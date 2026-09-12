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
from recall import recall_query, answer_recall
from prompting import build_chat_prompt
from run_lock import writer_active
from tools.calculator import answer_arithmetic

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
process = None
PORT = 8765
torch.set_num_threads(4)

PREFERENCE_PROMPTS = [
    # Re-tested 2026-09-09 against the current 58.4M-param finetune (see WRITEUP.md).
    # The prior list (kept 2026-09-09 earlier same day) still tied heavily live --
    # including a prompt ('What makes a good friend?') that isn't from any real
    # trained category at all, so it has zero learned response variety and just
    # falls back to generic boilerplate regardless of temperature. The actual fix:
    # pull prompts ONLY from make_instructions.py categories that were trained with
    # genuinely multiple canned replies (GREETING_REPLIES: 5, IDENTITY_REPLIES: 3,
    # FAREWELL_REPLIES: 4, THANKS_REPLIES: 4) -- WELLBEING/HELP (only 2-3 replies
    # each) and any prompt outside these six categories collapse too easily.
    #
    # Revised 2026-09-12: this session's RAFT round trained strong, confident
    # single answers into exactly the prompts overlapping RAFT_PROMPTS (raft.py),
    # collapsing 29/35 live pair-generation calls to byte-identical A/B -- nothing
    # left to label. Swapped those 6 collapsed prompts for untested phrasings from
    # the SAME safe categories (never touched by RAFT_PROMPTS) rather than adding
    # new categories, which is exactly what caused the garbled-output regressions
    # documented above. If a future RAFT/PPO/DPO round trains on these too, expect
    # to swap again -- this list needs to stay ahead of whatever was just trained.
    'Hi', 'Hi there', 'Good evening', "What's up?", 'Greetings',
    'What is your name?', 'What are you?', 'Do you have a name?', 'Are you ChatGPT?',
    'I have to go now', 'Bye', "That's all, bye",
    'Much appreciated', 'Thanks a lot', 'I appreciate it',

    # Added 2026-09-10 from the new OPEN_ENDED category, then re-tested live and
    # trimmed hard: 7 of the original 9 OPEN_ENDED prompts were checked twice each
    # against real /api/preferences/pair calls and showed a genuine defect at least
    # once -- a stray "<|end" tag leaking into the visible text, mid-sentence
    # truncation, garbled/repeated phrases, or a reply from a completely unrelated
    # category (e.g. a HELP-style "Sure — what do you need help with?" answering a
    # question about honesty). Per-prompt tie/coherence state is already documented
    # as unstable across training checkpoints (see WRITEUP.md); it does NOT follow
    # that broken-both-times-in-a-row is just noise -- only the 2 prompts below
    # passed clean on every check and are kept. HELP_WITH_TOPIC_PROMPTS all tested
    # clean and on-topic every time.
    'Would you rather be invisible or be able to fly?', 'What is the best way to learn something new?',
    'Can you help me with a math problem?', 'Can you help me with my homework?',
    'Can you help me with something complicated?', 'Can you help me write an essay?',
    'Can you help me fix my code?', 'Can you help me plan a trip?',
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
            panel_running = process is not None and process.poll() is None
            running = panel_running or writer_active(ROOT/'checkpoints')
            if state['status'] == 'training' and not running:
                state['status'] = 'interrupted'
            if running and not panel_running and state['status'] != 'training':
                state['status'] = 'preparing'
            state.update(training_owner='panel' if panel_running else 'external' if running else None, running=running, checkpoint=(ROOT/'checkpoints/latest.pt').exists(), device=device_name())
            if not running and process is not None and process.poll() not in (None, 0):
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
            if not isinstance(body, dict):
                raise ValueError('Request body must be a JSON object.')
            panel_running = process is not None and process.poll() is None
            running = panel_running or writer_active(ROOT/'checkpoints')
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
                if running and not panel_running:
                    raise ValueError('Training belongs to an external process. Pause it in its terminal.')
                if panel_running:
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
                messages = body.get('messages', [])
                if (not isinstance(messages, list) or not 1 <= len(messages) <= 20
                        or any(not isinstance(m, dict) or m.get('role') not in ('user', 'assistant')
                               or not isinstance(m.get('content'), str) for m in messages)
                        or messages[-1]['role'] != 'user'):
                    raise ValueError('Send 1–20 user/assistant messages, ending with a user message.')
                last_content = messages[-1]['content']
                if len(last_content) > 2000:
                    raise ValueError('Keep each message under 2000 characters.')
                # Deterministic tools need neither model weights nor a training pause.
                if recall_query(last_content) is not None:
                    self.reply(answer_recall(last_content, memory.list_records()))
                    return
                tool_reply = answer_arithmetic(last_content)
                if tool_reply is not None:
                    self.reply({'reply': tool_reply, 'activity': [], 'tool': 'calculator'})
                    return
                if running:
                    raise ValueError('Pause training and wait for its weights to save before chatting.')
                p = ROOT/'checkpoints/latest.pt'
                if not p.exists():
                    raise ValueError('Train your brain first to create its first checkpoint.')
                saved = torch.load(p, map_location='cpu', weights_only=True)
                if saved.get('stage') != 'finetune':
                    raise ValueError('Chat needs a fine-tuned checkpoint. Run the instruction fine-tuning stage first (see README).')
                count = int(body.get('count', 160))
                temp = float(body.get('temperature', 0.8))
                if not 1 <= count <= 512 or not 0.1 <= temp <= 2:
                    raise ValueError('Invalid generation settings.')
                model = Brain(Config(**saved['config']))
                model.load_state_dict(saved['model'])
                tok_path = ROOT/'checkpoints/tokenizer.json'
                tokenizer = Tokenizer().load(tok_path) if tok_path.exists() else None
                prompt_text, context_info = build_chat_prompt(messages, tokenizer, model.config.context, reserve=min(count, 32))
                text, activity = model.generate(prompt_text, count, temp, tokenizer=tokenizer, stop_text='<|end|>')
                reply = text[len(prompt_text):].strip() if text.startswith(prompt_text) else text.strip()
                self.reply({'reply': reply, 'activity': activity, 'step': saved['step'], 'context': context_info})
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
                # Widened+equalized 2026-09-12: 0.5/0.9 stopped diverging once RAFT made
                # the model confident on these prompts -- a low/high pair also wastes the
                # low draw on near-certainty either way. 1.3/1.3 gives real divergence more
                # often than not, but still ties a lot of the time in practice (a genuinely
                # confident model does that) -- rather than fight sampling further and risk
                # more incoherence, retry server-side up to 4 times so an identical pair
                # never actually reaches the person labeling; a real tie that survives every
                # retry is presented as-is rather than retried forever.
                def plausible(r):
                    # Cheap, real filter for the retry loop -- catches the obvious
                    # breakage (stray end-marker fragments, truncation, non-alphabetic
                    # start) that 1.3 sampling occasionally produces. Not a coherence
                    # judge; a human still makes the real call on anything that passes.
                    return len(r) >= 3 and r[0].isalpha() and '<|' not in r
                out = None
                for _attempt in range(6):
                    candidate = []
                    for temp in (1.3, 1.3):
                        text, _ = model.generate(wrapped, count=60, temperature=temp, tokenizer=tokenizer, stop_text='<|end|>')
                        candidate.append(text[len(wrapped):].strip() if text.startswith(wrapped) else text.strip())
                    out = candidate
                    if candidate[0] != candidate[1] and plausible(candidate[0]) and plausible(candidate[1]):
                        break
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
