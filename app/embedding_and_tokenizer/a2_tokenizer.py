from transformers import AutoModelForSequenceClassification, AutoTokenizer
from icecream import ic

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("/mnt/nas1/models/bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained(
    "finetune_demo/THUDM/chatglm3-6b", trust_remote_code=True
)

examples = {
    "s1": [
        "Generating train split: 0 examples [00:00, ? examples/s]",
        "Generating train split: 44644 examples [00:00, 302845.87 examples/s]",
    ],
    "s2": [
        "Generating validation split: 0 examples [00:00, ? examples/s]",
        "Generating validation split: 1070 examples [00:00, 178786.76 examples/s]",
    ],
}

r = tokenizer(examples["s1"], examples["s2"])

# dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
# <class 'list'> <class 'list'> <class 'int'>
ic(r.keys())
ic(type(r["input_ids"]), type(r["input_ids"][0]), type(r["input_ids"][0][0]))

r0 = tokenizer.get_command("[gMASK]")
r1 = tokenizer.get_command("sop")
ic(r0)
ic(r1)
ic(tokenizer.get_command("<|user|>"))
message = {
    "role": "user",
    "content": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤",
}
tokens = tokenizer.build_single_message(message["role"], "", message["content"])
ic(tokens)
# not add special tokens
# tokens: [30910, 33467, 31010, 56532, 30998, 55090, 54888, 31010, 40833, 30998, 32799, 31010, 40589, 30998, 37505, 31010, 37216, 30998, 56532, 54888, 31010, 56529, 56158, 56532]
tokens = tokenizer.tokenizer.encode(message["content"])
ic(tokens)
# not add special tokens
# [30910, 13]
tokens = tokenizer.tokenizer.encode("\n")
ic(tokens)

# default, add 2 special tokens: [gMASK], sop
# prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
tokens = tokenizer.encode(message["content"])
ic(tokens)
back_str = tokenizer.decode(tokens)
ic(back_str)
tokens = tokenizer.encode(message["content"], add_special_tokens=False)
ic(tokens)
"""  
tokenizer.special_tokens: {'<bos>': 1, '<eos>': 2, '<unk>': 0, '<pad>': 0}
tokenizer.tokenizer.special_tokens: {'[MASK]': 64789, '[gMASK]': 64790, '[sMASK]': 64791, 'sop': 64792, 'eop': 64793, '<|system|>': 64794, '<|user|>': 64795, '<|assistant|>': 64796, '<|observation|>': 64797}
"""
ic(tokenizer.special_tokens)
ic(tokenizer.tokenizer.special_tokens)

# direct_input_ids = tokenizer.encode()
## 30910 is the start token of a sentence, but not special tokens.
# token = tokenizer.convert_ids_to_tokens(30910)
# ic(token)
# s = tokenizer.convert_ids_to_tokens(tokens)
# ic(s)
# s = tokenizer.convert_tokens_to_string(s)
# ic(s)
# print(f'**{token}*_*')
# def build_single_message(self, role, metadata, message):
#     assert role in ["system", "user", "assistant", "observation"], role
#     role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
#     message_tokens = self.tokenizer.encode(message)
#     tokens = role_tokens + message_tokens
#     return tokens
