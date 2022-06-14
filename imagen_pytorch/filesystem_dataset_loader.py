import sqlite3
from sqlite3 import Error, Connection

from transformers import AutoTokenizer
import numpy as np
from PIL import Image
import io

def create_connection(db_file: str) -> Connection:
  """ create a database connection to a SQLite database """
  conn = None
  try:
    conn = sqlite3.connect(db_file)
    print(sqlite3.version)
  except Error as e:
    print(e)
  finally:
    if conn:
      conn.close()

class FilesystemDatasetReader:
  def __init__(
    self,
    tokenizer: AutoTokenizer,
    db_file = '/Users/birch/machine-learning/Safebooru 2022a/safebooru.db',
    # input_dataset,
    # batch_size,
    # num_prepro_workers,
    # enable_text=True,
    # enable_image=True,
    # enable_metadata=False,
    # wds_image_key="jpg",
    # wds_caption_key="txt",
    # cache_path=None,
  ):
    # self.batch_size = batch_size
    self.conn = create_connection(db_file)
    self.tokenizer = tokenizer
    dataset = self.create_dataset(
      input_dataset,
      enable_text=enable_text,
      enable_image=enable_image,
      image_key=wds_image_key,
      caption_key=wds_caption_key,
      enable_metadata=enable_metadata,
      cache_path=cache_path,
      t5_name=t5_name
    )
    self.dataloader = dataset_to_dataloader(
      dataset, batch_size, num_prepro_workers, "webdataset")

  def get_loader(self):
    return self.dataloader

  def __iter__(self):
    for batch in self.dataloader:
        yield batch

  def create_dataset(self):
    def tokenize(text):
      if np.random.binomial(1, 0.08):
        text = ''
      text_encoding = self.tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")

      tokens, *_ = text_encoding['input_ids']
      mask, *_ = text_encoding['attention_mask']

      return {
        "tokens": tokens,
        "mask": mask,
      }

    def preprocess_dataset(item):
      resolution = 64

      cur = self.conn.cursor()
      cur.execute("SELECT ")

      # image_data = 
      
      pil_image = Image.open(io.BytesIO(image_data))
      pil_image.load()
      while min(*pil_image.size) >= 2 * resolution:
          pil_image = pil_image.resize(
              tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
          )

      scale = resolution / min(*pil_image.size)
      pil_image = pil_image.resize(
          tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
      )

      arr = np.array(pil_image.convert("RGB"))
      crop_y = (arr.shape[0] - resolution) // 2
      crop_x = (arr.shape[1] - resolution) // 2
      
      arr = arr[crop_y: crop_y + resolution, crop_x: crop_x + resolution]
      arr = arr.astype(np.float32) / 127.5 - 1
          
      text = item[caption_key]
      caption = text.decode("utf-8")
      tokenized_text = tokenizer(caption)
      return np.transpose(arr, [2, 0, 1]), tokenized_text

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
