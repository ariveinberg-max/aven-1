import http.client
import json
import threading
import tempfile
from pathlib import Path
import torch
from brain import Brain, Config
from tokenizer import Tokenizer
import unittest
from unittest.mock import patch, Mock
import server


class ChatToolTests(unittest.TestCase):
    def test_tools_work_during_training_without_loading_weights(self):
        httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        worker = threading.Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.object(server, 'PORT', httpd.server_port), patch.object(server, 'process', Mock(poll=Mock(return_value=None))), patch.object(server.torch, 'load') as load, patch.object(server.memory, 'list_records', return_value=[dict(id=9, title='Engine', content='Use blue fuel.')]):
                for message, tool in [('(2+3)*4', 'calculator'), ('/recall engine', 'memory')]:
                    conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port)
                    conn.request('POST', '/api/chat', json.dumps({'messages':[{'role':'user','content':message}]}), {'Content-Type':'application/json'})
                    response = conn.getresponse()
                    result = json.loads(response.read())
                    conn.close()
                    self.assertEqual(response.status, 200, result)
                    self.assertEqual(result['tool'], tool)
                load.assert_not_called()
                conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port)
                conn.request('POST', '/api/chat', json.dumps({'messages':[None]}))
                response = conn.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
                conn.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join()


    def test_model_chat_context_metadata_and_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'checkpoints').mkdir()
            model = Brain(Config(width=32, layers=1, heads=4, context=128))
            torch.save(dict(config=vars(model.config), model=model.state_dict(), step=1,
                            stage='finetune'), root / 'checkpoints/latest.pt')
            Tokenizer().save(root / 'checkpoints/tokenizer.json')
            httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
            worker = threading.Thread(target=httpd.serve_forever, daemon=True)
            worker.start()
            try:
                with patch.object(server, 'ROOT', root), patch.object(server, 'PORT', httpd.server_port), patch.object(server, 'process', None):
                    for content, status in [('hello', 200), ('x' * 200, 400)]:
                        messages = [dict(role='user', content='old '*40),
                                    dict(role='assistant', content='old answer'),
                                    dict(role='user', content=content)]
                        conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port)
                        conn.request('POST', '/api/chat', json.dumps(dict(messages=messages, count=2)))
                        response = conn.getresponse()
                        result = json.loads(response.read())
                        conn.close()
                        self.assertEqual(response.status, status, result)
                        if status == 200:
                            self.assertEqual(result['context']['omitted_messages'], 2)
                            self.assertLessEqual(result['context']['prompt_tokens'], 128)
                        else:
                            self.assertIn('Shorten', result['error'])
            finally:
                httpd.shutdown()
                httpd.server_close()
                worker.join()


    def test_external_writer_is_visible_and_not_stopped_by_panel(self):
        from run_lock import WriterLock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
            worker = threading.Thread(target=httpd.serve_forever, daemon=True)
            worker.start()
            try:
                with patch.object(server, 'ROOT', root), patch.object(server, 'PORT', httpd.server_port), patch.object(server, 'process', None), WriterLock(root/'checkpoints'), patch.object(server.subprocess, 'Popen') as launch:
                    conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port)
                    conn.request('GET', '/api/status')
                    response = conn.getresponse()
                    result = json.loads(response.read())
                    self.assertTrue(result['running'])
                    self.assertEqual(result['training_owner'], 'external')
                    self.assertEqual(result['status'], 'preparing')
                    conn.close()
                    for endpoint in ['train', 'stop']:
                        conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port)
                        conn.request('POST', '/api/' + endpoint, '{}')
                        response = conn.getresponse()
                        self.assertEqual(response.status, 400)
                        response.read()
                        conn.close()
                    launch.assert_not_called()
            finally:
                httpd.shutdown()
                httpd.server_close()
                worker.join()
