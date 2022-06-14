from imagen_pytorch.train_utils import TrainLoop
from imagen_pytorch.train_all import _fix_path
from imagen_pytorch.model_creation import model_and_diffusion_defaults
from imagen_pytorch.model_creation import create_model_and_diffusion

from transformers import AutoTokenizer
from imagen_pytorch.filesystem_dataset_loader import FilesystemDatasetReader

def main():
  options = model_and_diffusion_defaults()
  options['use_fp16'] = False
  options['t5_name'] = 't5-3b'
  options['cache_text_emb'] = False
  options['cache_text_emb'] = True
  model, diffusion = create_model_and_diffusion(**options)

  checkpoint_path = './ImagenT5-3B/model.pt'
  model.load_state_dict(_fix_path(checkpoint_path), strict=False)

  tokenizer = AutoTokenizer.from_pretrained('t5-3b')
  reader = FilesystemDatasetReader(
    tokenizer,
  )

  TrainLoop(
    model=model,
    diffusion=diffusion,
    data=reader,
    batch_size=args.batch_size,
    microbatch=-1,
    lr=1e-4,
    ema_rate="0.9999",
    log_interval=100,
    save_interval=args.save_interval,
    resume_checkpoint=False,
    use_fp16=False,
    fp16_scale_growth=1e-3,
    weight_decay=0.01,
    lr_anneal_steps=0,
    save_dir=args.save_dir,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps
  ).run_loop()

if __name__ == '__main__':
  main()