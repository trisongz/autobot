import torch
import tensorflow as tf
import os
import logging
import random
import subprocess
import simdjson as json

from auto.utils.logger import get_logger
from torch.utils.data import Dataset

glob = tf.io.gfile.glob
logger = get_logger()


logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class LineSeekableFile:
    def __init__(self, seekable):
        self.fin = seekable
        self.line_map = list()
        self.line_map.append(0)
        while seekable.readline():
            self.line_map.append(seekable.tell())

    def __getitem__(self, index):
        self.fin.seek(self.line_map[index])
        return self.fin.readline()
    
    def __len__(self):
        return len(self.line_map)

class AutoDataset(Dataset):
    def __init__(self, input_files, tokenizer, input_field_name='text', input_max_length=1024, input_min_length=500, target_field_name=None, target_max_length=None, target_label_name='labels', seed=1, shuffle_files=True, debug=False, **kwargs):
        super().__init__()
        self.files = []
        self.debug = debug
        self.setup_files(input_files)
        if shuffle_files:
            random.seed(seed)
            random.shuffle(self.files)
        self.tokenizer = tokenizer
        self.params = {
            'src_name': input_field_name,
            'src_max': input_max_length,
            'src_min': input_min_length,
            'tgt_name': target_field_name,
            'tgt_max': target_max_length,
            'label': target_label_name
        }
        self.is_encdec = bool(target_field_name)
        self.token_cache = {
            'encoder': [],
            'encoder_attn': [],
            'decoder': [],
            'decoder_attn': []
        }
        self.text_chunks = ''
        self.sep_token = tokenizer.eos_token_id
        self.sep_word = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token_id
        self.parser = json.Parser()
        
    def setup_files(self, input_files):
        if isinstance(input_files, str):
            if input_files.endswith('*'):
                self.files = glob(input_files)
            elif os.path.isdir(input_files):
                self.files = glob(os.path.join(input_files, '*'))
        elif isinstance(input_files, list):
            for file_path in input_files:
                if os.path.isfile(file_path) and os.path.exists(file_path):
                    self.files.append(file_path)
                elif file_path.endswith('*'):
                    self.files.extend(glob(file_path))
                elif os.path.isdir(file_path):
                    self.files.extend(glob(os.path.join(file_path, '*')))
        
        self.total_files = len(self.files)
        self.file_idx, self.total_lines = {}, 0
        for x, file_path in enumerate(self.files):
            #total_lines = self.total_lines_in_file(file_path)
            self.file_idx[x] = {
                'start': self.total_lines, 'stop': 0, 'file': file_path, 
                'reader': LineSeekableFile(tf.io.gfile.GFile(file_path, 'rb'))
                }
            total_lines = len(self.file_idx[x]['reader'])
            self.file_idx['stop'] = (self.total_lines + total_lines)
            if self.debug:
                logger.info(f'File IDX Start: {self.total_lines} - File IDX End: {self.total_lines + total_lines}')
            self.total_lines += total_lines
        if self.debug:
            logger.info(f'Total Files: {self.total_files}. Total Lines: {self.total_lines}')
    
    def get_file_line(self, idx):
        for x in range(len(self.files)):
            if idx in range(self.file_idx[x]['start'], self.file_idx[x]['stop']):
                fidx = idx - self.file_idx[x]['start']
                if self.debug:
                    logger.info(f'File IDX: {fidx}')
                return self.file_idx[x]['reader'][fidx]

    def parse_json(self, line):
        try:
            return self.parser.parse(line).as_dict()
        except ValueError:
            return line
        except TypeError:
            return line            

    @classmethod
    def total_lines_in_file(cls, file_path):
        return int(subprocess.check_output(['wc', '-l', file_path]).split()[0])
    
    def tokenize_encoder(self, ex):
        return self.tokenizer(ex[self.params['src_name']])

    def tokenize_decoder(self, ex):
        return self.tokenizer(ex[self.params['tgt_name']])

    def tokenize_example(self, ex):
        tokens = self.tokenize_encoder(ex)
        result = {}
        if not self.is_encdec:
            if self.token_cache['encoder']: 
                if len(self.token_cache['encoder']) > self.params['src_max']:
                    result['input_ids'] = self.token_cache['encoder'][0:self.params['src_max']]
                    result['attention_mask'] = self.token_cache['encoder_attn'][0:self.params['src_max']]
                    self.token_cache['encoder'] = self.token_cache['encoder'][0:self.params['src_max']].extend(tokens['input_ids'].append(self.sep_token))
                    self.token_cache['encoder_attn'] = self.token_cache['encoder_attn'][0:self.params['src_max']].extend(tokens['attention_mask'].append(1))
                    
                else:
                    result['input_ids'] = self.token_cache['encoder'][:]
                    result['attention_mask'] = self.token_cache['encoder_attn'][:]

                    _to_slice = self.params['src_max'] - len(result['input_ids'])
                    result['input_ids'].extend(tokens['input_ids'][:_to_slice])
                    result['attention_mask'].extend(tokens['attention_mask'][:_to_slice])
                    self.token_cache['encoder'] = tokens['input_ids'][_to_slice:].append(self.sep_token) if tokens['input_ids'][_to_slice:] else []
                    self.token_cache['encoder_attn'] = tokens['attention_mask'][_to_slice:].append(1) if tokens['attention_mask'][_to_slice:] else []
                    
            else:
                result['input_ids'] = tokens['input_ids'][:self.params['src_max']]
                result['attention_mask'] = tokens['attention_mask'][:self.params['src_max']]
                self.token_cache['encoder'] = tokens['input_ids'][self.params['src_max']:].append(self.sep_token) if tokens['input_ids'][self.params['src_max']:] else []
                self.token_cache['encoder_attn'] = tokens['attention_mask'][self.params['src_max']:].append(1) if tokens['attention_mask'][self.params['src_max']:] else []
            
            if len(result['input_ids']) < self.params['src_max']:
                _to_pad = self.params['src_max'] - len(result['input_ids'])
                result['input_ids'].extend([self.pad_token for i in range(_to_pad)])
                result['attention_mask'].extend([1 for i in range(_to_pad)])
            
            result[self.params['label']] = [-100 if token == self.pad_token else token for token in result['input_ids']]

        else:
            dtokens = self.tokenize_decoder(ex)
            result['input_ids'] = tokens['input_ids'][:self.params['src_max']]
            result['attention_mask'] = tokens['attention_mask'][:self.params['src_max']]
            result[self.params['label']] = dtokens['input_ids'][:self.params['tgt_max']]
            result['decoder_attention_mask'] = dtokens['attention_mask'][:self.params['tgt_max']]
            if len(result[self.params['label']]) < self.params['tgt_max']:
                _to_pad = self.params['tgt_max'] - len(result[self.params['label']])
                result[self.params['label']].extend([-100 for i in range(_to_pad)])
                result['decoder_attention_mask'].extend([1 for i in range(_to_pad)])

            result[self.params['label']] = [-100 if token == self.pad_token else token for token in result[self.params['label']]]

        for k,v in result.items():
            result[k] = torch.tensor(v, dtype=torch.long)
        
        return result

    def _get_example(self, idx):
        idx = idx if idx <= self.total_lines else random.randint(0, self.total_lines)
        ex = self.get_file_line(idx)
        if not ex:
            while True:
                new_idx = random.randint(0, self.total_lines)
                if self.debug:
                    logger.info(f'Bad IDX: {idx} - New Random IDX: {new_idx}')
                ex = self.get_file_line(new_idx)
                if ex:
                    break
        return self.parse_json(ex.strip())

    def _seq_len(self, ex):
        return len(ex[self.params['src_name']].split()) > self.params['src_max']
    
    def __getitem__(self, idx):
        if self.debug:
            logger.info(f'Getting IDX: {idx}')
        ex = self._get_example(idx)
        if self._seq_len(ex) or self.is_encdec:
            return self.tokenize_example(ex)
        else:
            cidx = 1
            while True:
                self.text_chunks = self.text_chunks + (ex[self.params['src_name']] + ' ' + self.sep_word)
                if self._seq_len({self.params['src_name']: self.text_chunks}):
                    result = self.tokenize_example({self.params['src_name']: self.text_chunks})
                    self.text_chunks = ''
                    break
                ex = self._get_example(idx+cidx)
                cidx += 1
            return result

    def __len__(self):
        return self.total_lines