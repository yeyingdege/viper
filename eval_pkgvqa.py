import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from decord_func import decord_video_given_start_end_seconds
from utils import parse_choice
from main_simple_lib import *

from logger import setup_logger

logger = setup_logger("vipergpt", ".", 0, filename="vipergpt.log")


class TypeAccuracy(object):
    def __init__(self, type_name):
        self.correct = 0
        self.total = 10e-9
        self.type_name = type_name

    def update(self, gt, pred):
        self.total += 1
        if "{}".format(pred) in gt:
            self.correct += 1

    def get_accuracy(self):
        return 1.0*self.correct / self.total

    def print_accuracy(self):
        #print(f"{self.type_name} Accuracy: {self.get_accuracy()} | {self.correct}/{self.total}")
        print("{} Accuracy: {:.4f} | {}/{}".format(
                self.type_name,
                self.get_accuracy(),
                self.correct,
                int(self.total)
            ))



QUESTION_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
                  'qa4_step','qa5_task', 'qa6_precedingStep', 'qa7_bestPrecedingStep',
                  'qa8_toolNextStep', 'qa9_bestInitial','qa10_bestFinal', 'qa11_domain',
                  'qa12_toolPurpose', 'qa13_actionPurpose', 'qa14_objectPurpose',
                  'qa15_ToolOtherPurpose', 'qa16_ObjectOtherPurpose', 'qa17_AlternativeTool',
                  'qa18_TaskSameToolSamePurpose', 'qa19_TaskSameObjectSamePurpose']
# SKIP_EVAL_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
#                    'qa4_step','qa5_task', 'qa6_precedingStep', 'qa7_bestPrecedingStep',
#                    'qa9_bestInitial','qa10_bestFinal', 'qa11_domain']

def main(args):
    vipergpt_error_cnt = 0
    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Overall Accuracy for All Questions
    global_acc = TypeAccuracy("Global")
    qa_acc = []
    for t in range(len(QUESTION_TYPES)):
        qa_acc.append(TypeAccuracy(f"qa{t+1}_"))


    total = 0
    results = {}
    for line in tqdm(annotations, total=len(annotations)):
        # Q-A Pair
        idx = line["qid"]
        quest_type = line["quest_type"]
        conversations = line["conversations"]
        qs = conversations[0]["value"]
        gt_answers = conversations[1]["value"]
        results[idx] = {"qid": idx, "quest_type": quest_type, 
                        "qs": qs, "gt": gt_answers,
                        "task_label": line["task_label"], 
                        "step_label": line["step_label"]}
        
        with torch.inference_mode():
            if args.num_video_frames > 0:
                # Load Image
                video_path = os.path.join(args.image_folder, line["video"])

                if "start_secs" in line:
                    start_secs = line['start_secs']
                    end_secs = line['end_secs']
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path, 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=args.num_video_frames)
                else:
                    frames, frame_indices =  decord_video_given_start_end_seconds(video_path,
                        num_video_frames=args.num_video_frames)

                images =[ Image.fromarray(x).convert('RGB') for x in frames ]
                n_images = len(images)

                image = transforms.ToTensor()(images[0])
                qs = qs.replace("<video>\n", "")
        logger.info(f"{idx}\nquestion:{qs}\nanswer:{gt_answers}")
        total += 1
        # Decode output
        try:
            code = get_code(qs)
            logger.info(f"code:\n{code}")
            outputs = execute_code(code, image, show_intermediate_steps=True)
        except:
            vipergpt_error_cnt += 1
            outputs = None
            print('vipergpt encountered error')
        logger.info(f"output:\n{outputs}")
        outputs = str(outputs)

        answer_id = parse_choice(outputs, line["all_choices"], line["index2ans"])
        results[idx]["response"] = outputs
        results[idx]["parser"] = answer_id
        # print("qid {}:\n{}".format(idx, qs))
        print("AI: {}\nParser: {}\nGT: {}\n".format(outputs, answer_id, gt_answers))

        global_acc.update(gt_answers, answer_id)
        for t in range(len(QUESTION_TYPES)):
            if f"qa{t+1}_" in quest_type:
                qa_acc[t].update(gt_answers, answer_id)

        # print each type accuracy
        print("-----"*5)
        acc_list = []
        for t in range(len(QUESTION_TYPES)):
            qa_acc[t].print_accuracy()
            acc_list.append(qa_acc[t].get_accuracy())
        global_acc.print_accuracy()
        print("-----"*5)
        avg_acc = sum(acc_list) / len(acc_list)
        print("Average Acc over Type: {:.4f}".format(avg_acc))
        logger.info(f"global_acc: {global_acc.print_accuracy()}, avg_acc: {avg_acc}")

    # save all results
    print("save to {}".format(args.answers_file))
    with open(args.answers_file, "w") as f:
        json.dump(results, f, indent=2)

    print('vipergpt_error_cnt', vipergpt_error_cnt)
    print("Process Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="data/COIN/videos")
    parser.add_argument("--question-file", type=str, default="data/testing_vqa19_25oct_v2.json")
    parser.add_argument("--answers-file", type=str, default="data/answers_vipergpt_f1_25oct.json")
    parser.add_argument("--num_video_frames", type=int, default=1)
    #parser.add_argument("--tokenizer_model_max_length", type=int, default=8192)
    args = parser.parse_args()
    main(args)
