from transformers import AutoModelForSequenceClassification, AutoTokenizer
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("/mnt/nas1/models/bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained('finetune_demo/THUDM/chatglm3-6b', trust_remote_code=True)

examples = {
    's1': ['Generating train split: 0 examples [00:00, ? examples/s]', 'Generating train split: 44644 examples [00:00, 302845.87 examples/s]'],
    's2': ['Generating validation split: 0 examples [00:00, ? examples/s]', 'Generating validation split: 1070 examples [00:00, 178786.76 examples/s]']
}

r = tokenizer(examples['s1'], examples['s2'])

# dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
# <class 'list'> <class 'list'> <class 'int'>
ic(r.keys())
ic(type(r['input_ids']), type(r['input_ids'][0]), type(r['input_ids'][0][0]))

r0 = tokenizer.get_command('[gMASK]')
r1 = tokenizer.get_command('sop')
ic(r0)
ic(r1)
message = {'role': 'user', 'content': '类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤'}
new_input_ids = tokenizer.build_single_message(
    message['role'], '', message['content']
)
ic(new_input_ids)
# direct_input_ids = tokenizer.encode()