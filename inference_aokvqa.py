from magma import Magma
from magma.image_input import ImageInput
from path import Path
import json
 
question_file = "/gpfs/alpine/csc499/proj-shared/magma/aokvqa/aokvqa_v1p0_val.json"
image_directory = "/gpfs/alpine/csc499/proj-shared/magma/coco17/val2017"

# question_file = "/gpfs/alpine/csc499/proj-shared/magma/okvqa/OpenEnded_mscoco_val2014_questions.json"
# image_directory = "/gpfs/alpine/csc499/proj-shared/magma/coco/val2014"

# Opening JSON file
with open(question_file) as json_file:
    data = json.load(json_file)
    questions = data
    # questions = data["questions"]

    # checkpoint = "/gpfs/alpine/csc499/proj-shared/magma/checkpoints/alexisroger/original_MAGMA.pt"
    checkpoint = "/gpfs/alpine/scratch/alexisroger/csc499/magma/checkpoints/MAGMA_160M_clipH_10/global_step16000/mp_rank_00_model_states.pt"

    checkpoint = Path(checkpoint)

    model = Magma.from_checkpoint(
                config_path = "summit_clipH_pythia_70m_forward.yml",
                checkpoint_path = checkpoint,
                device = "cuda"
            )

    responses = {}

    for question in questions:
        print(question)
        # {"image_id": 297147, "question": "What sport can you use this for?", "question_id": 2971475}
        img_num, question_id, question = int(question["image_id"]), question["question_id"], question["question"]

        inputs =[
            ## supports urls and path/to/image
            # ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
            ImageInput(f'{image_directory}/{img_num:012d}.jpg'),
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

        responses[str(question_id)] = {
            'direct_answer' : res
        }

    with open("test.json", "w") as outfile:
        json.dump(responses, outfile)
