"""Loopback-only local training panel. No cloud services or model downloads."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import webbrowser
import torch
from brain import Brain, Config, device_name
from tokenizer import Tokenizer
import memory
import sources
import preferences
import random
from tools.calculator import answer_arithmetic

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
process = None
PORT = 8765
torch.set_num_threads(4)

PREFERENCE_PROMPTS = [
    # Re-tested 2026-09-09 against the current 58.4M-param finetune (see WRITEUP.md).
    # Two live 0.5/0.9-temperature runs of the full old list showed the tie rate had
    # dropped from 86% (old 19.8M model) to ~53%, but not because every prompt got
    # better: some (identity/farewell prompts like 'Introduce yourself', 'Who are you?',
    # 'Can you help me?', 'Goodbye', 'See you later', 'What do you think about school?',
    # 'How are you doing?') now have ONE dominant canned phrasing and tied in both runs
    # -- dropped, same over-drilled-category problem as arithmetic/antonyms before.
    # Others ('Hey there', 'Would you rather be invisible or be able to fly?', 'Is it
    # better to be cautious or take risks?') did vary across temperatures, but into
    # incoherent non-sequiturs ("There are twelve months in a year.", "a bednac") rather
    # than genuinely different valid answers -- that's noise, not preference signal, so
    # dropped too. What's left is the prompts that showed real, coherent multi-reply
    # variance in both test runs.
    'Tell me about yourself', 'What kind of AI are you?',
    'Hello!', 'Good morning', "How's it going?",
    'Thank you', 'Thanks for the help', 'I have a question',
    'What makes a good friend?',
]


class Handler(BaseHTTPRequestHandler):
    def reply(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def trusted(self):
        origin = self.headers.get('Origin')
        return self.headers.get('Host') in (f'127.0.0.1:{PORT}', f'localhost:{PORT}') and origin in (None, f'http://127.0.0.1:{PORT}', f'http://localhost:{PORT}')

    def do_GET(self):
        if not self.trusted():
            return self.reply({'error': 'Local access only'}, 403)
        if self.path in ('/', '/chat'):
            page = 'chat.html' if self.path == '/chat' else 'ui.html'
            body = (ROOT/page).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/status':
            p = ROOT/'checkpoints/status.json'
            state = json.loads(p.read_text()) if p.exists() else {'status': 'ready', 'step': 0, 'history': []}
            running = process is not None and process.poll() is None
            if state['status'] == 'training' and not running:
                state['status'] = 'interrupted'
            state.update(running=running, checkpoint=(ROOT/'checkpoints/latest.pt').exists(), device=device_name())
            if process is not None and process.poll() not in (None, 0):
                state.update(status='error', error=(ROOT/'work/train.log').read_text()[-2000:])
            return self.reply(state)
        elif self.path == '/api/memory':
            try:
                self.reply({'records': memory.list_records()})
            except Exception as exc:
                self.reply({'error': str(exc)}, 500)
        elif self.path == '/api/sources':
            try:
                items = sources.list_sources()
                self.reply({'sources': items, 'corpus_locked': (ROOT/'checkpoints/latest.pt').exists()})
            except Exception as exc:
                self.reply({'error': str(exc)}, 500)
        elif self.path == '/api/preferences':
            try:
                self.reply({'records': preferences.list_recent(), 'count': preferences.count()})
            except Exception as exc:
                self.reply({'error': str(exc)}, 500)
        else:
            self.reply({'error': 'Not found'}, 404)

    def do_POST(self):
        global process
        if not self.trusted():
            return self.reply({'error': 'Local access only'}, 403)
        if not LOCK.acquire(blocking=False):
            return self.reply({'error': 'Another operation is running. Try again shortly.'}, 409)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size < 0 or size > 1_000_000:
                raise ValueError('Request too large. Use the data folder for larger text files.')
            body = json.loads(self.rfile.read(size) or b'{}')
            running = process is not None and process.poll() is None
            if self.path == '/api/train':
                if running:
                    raise ValueError('Training is already running.')
                steps = int(body.get('steps', 200))
                if not 1 <= steps <= 2000:
                    raise ValueError('Choose 1–2000 additional steps.')
                dataset = ROOT/'data/training.txt'
                if not dataset.exists():
                    dataset = ROOT/'data/demo.txt'
                (ROOT/'work').mkdir(exist_ok=True)
                cmd = [sys.executable, str(ROOT/'train.py'), '--data', str(dataset), '--steps', str(steps), '--resume']
                netrc = Path.home()/'.netrc'
                if netrc.exists() and 'api.wandb.ai' in netrc.read_text(encoding='utf-8', errors='ignore'):
                    cmd.append('--wandb')
                with (ROOT/'work/train.log').open('w') as log:
                    process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                self.reply({'ok': True})
            elif self.path == '/api/stop':
                if running:
                    process.terminate()
                self.reply({'ok': True})
            elif self.path == '/api/data':
                if running or (ROOT/'checkpoints/latest.pt').exists():
                    raise ValueError('This run already has weights. Keep its corpus unchanged; see README for starting a new run.')
                raw = str(body.get('text', '')).encode('utf-8')
                if len(raw) < 4096:
                    raise ValueError('Add at least 4 KB of text (roughly 700–1000 words).')
                (ROOT/'data/training.txt').write_bytes(raw)
                self.reply({'ok': True, 'bytes': len(raw)})
            elif self.path == '/api/generate':
                if running:
                    raise ValueError('Pause training and wait for its weights to save before generating.')
                p = ROOT/'checkpoints/latest.pt'
                if not p.exists():
                    raise ValueError('Train your brain first to create its first checkpoint.')
                prompt = str(body.get('prompt', 'Once upon a time'))
                if len(prompt.encode()) > 4096:
                    raise ValueError('Keep the prompt under 4096 bytes.')
                count = int(body.get('count', 160))
                temp = float(body.get('temperature', 0.8))
                if not 1 <= count <= 512 or not 0.1 <= temp <= 2:
                    raise ValueError('Invalid generation settings.')
                saved = torch.load(p, map_location='cpu', weights_only=True)
                model = Brain(Config(**saved['config']))
                model.load_state_dict(saved['model'])
                tok_path = ROOT/'checkpoints/tokenizer.json'
                tokenizer = Tokenizer().load(tok_path) if tok_path.exists() else None
                finetuned = saved.get('stage') == 'finetune'
                model_prompt = f'### Instruction:\n{prompt}\n\n### Response:\n' if finetuned else prompt
                text, activity = model.generate(model_prompt, count, temp, tokenizer=tokenizer, stop_text='<|end|>' if finetuned else None)
                if finetuned and text.startswith(model_prompt):
                    text = prompt + '\n' + text[len(model_prompt):]
                self.reply({'text': text, 'activity': activity, 'step': saved['step'], 'stage': saved.get('stage', 'pretrain')})
            elif self.path == '/api/chat':
                if running:
                    raise ValueError('Pause training and wait for its weights to save before chatting.')
                p = ROOT/'checkpoints/latest.pt'
                if not p.exists():
                    raise ValueError('Train your brain first to create its first checkpoint.')
                saved = torch.load(p, map_location='cpu', weights_only=True)
                if saved.get('stage') != 'finetune':
                    raise ValueError('Chat needs a fine-tuned checkpoint. Run the instruction fine-tuning stage first (see README).')
                messages = body.get('messages', [])
                if not isinstance(messages, list) or not messages:
                    raise ValueError('Send at least one message.')
                if len(messages) > 20:
                    raise ValueError('Keep conversations to 20 messages or fewer per request; start a new chat.')
                if messages[-1].get('role') != 'user':
                    raise ValueError('The last message must be from the user.')
                count = int(body.get('count', 160))
                temp = float(body.get('temperature', 0.8))
                if not 1 <= count <= 512 or not 0.1 <= temp <= 2:
                    raise ValueError('Invalid generation settings.')
                parts = []
                for m in messages[:-1]:
                    content = str(m.get('content', ''))[:2000]
                    if m.get('role') == 'user':
                        parts.append(f'### Instruction:\n{content}\n\n')
                    elif m.get('role') == 'assistant':
                        parts.append(f'### Response:\n{content}\n<|end|>\n\n')
                last_content = str(messages[-1].get('content', ''))[:2000]
                tool_reply = answer_arithmetic(last_content)
                if tool_reply is not None:
                    # Arithmetic is a documented model-level weakness (see
                    # WRITEUP.md, "Small models memorize arithmetic; they
                    # don't compute it") -- answer with real computation
                    # instead of the model's unreliable pattern-matched guess.
                    self.reply({'reply': tool_reply, 'activity': [], 'step': saved['step'], 'tool': 'calculator'})
                    return
                prompt_text = ''.join(parts) + f'### Instruction:\n{last_content}\n\n### Response:\n'
                model = Brain(Config(**saved['config']))
                model.load_state_dict(saved['model'])
                tok_path = ROOT/'checkpoints/tokenizer.json'
                tokenizer = Tokenizer().load(tok_path) if tok_path.exists() else None
                text, activity = model.generate(prompt_text, count, temp, tokenizer=tokenizer, stop_text='<|end|>')
                reply = text[len(prompt_text):].strip() if text.startswith(prompt_text) else text.strip()
                self.reply({'reply': reply, 'activity': activity, 'step': saved['step']})
            elif self.path == '/api/preferences/pair':
                p = ROOT/'checkpoints/latest.pt'
                if running:
                    raise ValueError('Pause training and wait for its weights to save before generating comparisons.')
                if not p.exists():
                    raise ValueError('Train your brain first to create its first checkpoint.')
                saved = torch.load(p, map_location='cpu', weights_only=True)
                if saved.get('stage') != 'finetune':
                    raise ValueError('Preference collection needs a fine-tuned checkpoint.')
                prompt = str(body.get('prompt', '')).strip()[:2000] or random.choice(PREFERENCE_PROMPTS)
                model = Brain(Config(**saved['config']))
                model.load_state_dict(saved['model'])
                tok_path = ROOT/'checkpoints/tokenizer.json'
                tokenizer = Tokenizer().load(tok_path) if tok_path.exists() else None
                wrapped = f'### Instruction:\n{prompt}\n\n### Response:\n'
                out = []
                for temp in (0.5, 0.9):
                    text, _ = model.generate(wrapped, count=60, temperature=temp, tokenizer=tokenizer, stop_text='<|end|>')
                    out.append(text[len(wrapped):].strip() if text.startswith(wrapped) else text.strip())
                self.reply({'prompt': prompt, 'response_a': out[0], 'response_b': out[1]})
            elif self.path == '/api/preferences/vote':
                prompt = str(body.get('prompt', ''))
                response_a = str(body.get('response_a', ''))
                response_b = str(body.get('response_b', ''))
                winner = str(body.get('winner', ''))
                if not prompt or not response_a or not response_b:
                    raise ValueError('Missing prompt or responses.')
                record_id = preferences.add(prompt, response_a, response_b, winner)
                self.reply({'ok': True, 'id': record_id, 'count': preferences.count()})
            elif self.path == '/api/preferences/delete':
                preferences.delete(int(body['id']))
                self.reply({'ok': True})
            elif self.path == '/api/memory/add':
                record_id = memory.add_record(str(body.get('title', '')), str(body.get('content', '')))
                self.reply({'ok': True, 'id': record_id})
            elif self.path == '/api/memory/update':
                memory.update_record(int(body['id']), str(body.get('title', '')), str(body.get('content', '')))
                self.reply({'ok': True})
            elif self.path == '/api/memory/delete':
                memory.delete_record(int(body['id']))
                self.reply({'ok': True})
            elif self.path == '/api/sources/read':
                self.reply({'text': sources.read_source(str(body.get('name', '')))})
            elif self.path == '/api/sources/save':
                if (ROOT/'checkpoints/latest.pt').exists() or running:
                    raise ValueError('Weights already exist for this run. Preserve checkpoints/ and start a new run to use a different corpus.')
                sources.add_or_update_source(str(body.get('name', '')), str(body.get('text', '')))
                self.reply({'ok': True})
            elif self.path == '/api/sources/delete':
                if (ROOT/'checkpoints/latest.pt').exists() or running:
                    raise ValueError('Weights already exist for this run. Preserve checkpoints/ and start a new run to use a different corpus.')
                sources.delete_source(str(body.get('name', '')))
                self.reply({'ok': True})
            elif self.path == '/api/sources/compile':
                if running or (ROOT/'checkpoints/latest.pt').exists():
                    raise ValueError('This run already has weights. Keep its corpus unchanged; see README for starting a new run.')
                written = sources.compile_corpus()
                self.reply({'ok': True, 'bytes': written})
            else:
                self.reply({'error': 'Not found'}, 404)
        except (ValueError, TypeError, KeyError) as exc:
            self.reply({'error': str(exc)}, 400)
        except Exception as exc:
            self.reply({'error': str(exc)}, 500)
        finally:
            LOCK.release()


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f'Your AI brain: http://127.0.0.1:{PORT}', flush=True)
    if '--open' in sys.argv:
        webbrowser.open(f'http://127.0.0.1:{PORT}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        server.server_close()
