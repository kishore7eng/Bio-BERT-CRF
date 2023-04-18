
from transformers import AutoTokenizer, AutoConfig, AutoModel



tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
config = AutoConfig.from_pretrained('dmis-lab/biobert-v1.1', num_labels=3)
model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1',config = config)


model.save_pretrained("/workspace/amit_pg/BioBertCRF/model")
tokenizer.save_pretrained('/workspace/amit_pg/BioBertCRF/tokenizer')
config.save_pretrained('/workspace/amit_pg/BioBertCRF/config')

