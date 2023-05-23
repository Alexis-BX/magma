from magma import Magma
from magma.image_input import ImageInput
from pathlib import Path
import json

prompts = [
    ("This is a photo of a ", "https://www.ikea.com/ca/en/images/products/faergklar-mug-matte-green__0986755_pe817319_s5.jpg"),
    ("This is a photo of a ", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"),
    ("This is a photo of a ", "https://media.zenfs.com/fr/sciencesetavenir.fr/7e0938020eda63afa6d363500ffb6c32"),
    ("This is a photo of a ", "https://upload.wikimedia.org/wikipedia/commons/3/34/Anser_anser_1_%28Piotr_Kuczynski%29.jpg"),
]

# checkpoint = "/gpfs/alpine/csc499/proj-shared/magma/checkpoints/alexisroger/original_MAGMA.pt"
# checkpoint = "/gpfs/alpine/scratch/alexisroger/csc499/magma/checkpoints/MAGMA_70M_clipH_9/global_step77500/mp_rank_00_model_states.pt"
#checkpoint = "/gpfs/alpine/scratch/alexisroger/csc499/magma/checkpoints/MAGMA_160M_clipH_10/global_step16000/mp_rank_00_model_states.pt"

checkpoint = "/gpfs/alpine/csc499/proj-shared/magma/checkpoints/alexisroger/MAGMA_160M_clipH_10/mp_rank_00_model_states_step2000.pt"

checkpoint = "/gpfs/alpine/csc499/proj-shared/magma/checkpoints/alexisroger/MAGMA_160M_clipH_10/mp_rank_00_model_states_step16000.pt"

checkpoint = "/gpfs/alpine/csc499/proj-shared/magma/checkpoints/alexisroger/MAGMA_160M_clipH_10/mp_rank_00_model_states_step58000.pt"


checkpoint = Path(checkpoint)

model = Magma.from_checkpoint(
            config_path = "summit_clipH_pythia_70m_forward.yml",
            checkpoint_path = checkpoint,
            device = "cuda"
        )

for prompt in prompts:
    question, image = prompt[0], prompt[1]

    inputs =[
        ## supports urls and path/to/image
        # ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
        ImageInput(image),
        question
    ]

    ## returns a tensor of shape: (1, 149, 4096)
    embeddings = model.preprocess_inputs(inputs)

    ## returns a list of length embeddings.shape[0] (batch size)
    res = ""
    while res == "":
        output = model.generate(
            embeddings = embeddings,
            max_steps = 6,
            temperature = 0.7,
            top_k = 0,
        )

        res = output[0].strip()

    print(res)
