import argparse
import torch
import os
import json
from tqdm import tqdm
from decord_func import decord_video_given_start_end_seconds
from utils import parse_choice, TypeAccuracy
from main_simple_lib import *
from vision_processes import queues_in
from datetime import datetime
from logger import setup_logger

logger = setup_logger("vipergpt", ".", 0, filename="results/vipergpt_1.log")


QUESTION_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
                  'qa4_step','qa5_task', 'qa6_precedingStep', 'qa7_bestPrecedingStep',
                  'qa8_toolNextStep', 'qa9_bestInitial','qa10_bestFinal', 'qa11_domain',
                  'qa12_toolPurpose', 'qa13_actionPurpose', 'qa14_objectPurpose',
                  'qa15_ToolOtherPurpose', 'qa16_ObjectOtherPurpose', 'qa17_AlternativeTool',
                  'qa18_TaskSameToolSamePurpose', 'qa19_TaskSameObjectSamePurpose']
SKIP_EVAL_TYPES = ['qa4_step','qa5_task']


def main(args):
    st = datetime.now()
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
    # load past evaluation results, supporting multiple evaluation without repetition
    if os.path.exists(args.answers_file):
        results = json.load(open(args.answers_file, "r"))
    for i, line in tqdm(enumerate(annotations), total=len(annotations)):
        # Q-A Pair
        qid = line["qid"]
        # skip evaluating examples with existing results
        if qid in results:
            continue
        quest_type = line["quest_type"]
        if quest_type in SKIP_EVAL_TYPES:
            continue
        conversations = line["conversations"]
        qs = conversations[0]["value"]
        gt_answers = conversations[1]["value"]
        results[qid] = {"qid": qid, "quest_type": quest_type, 
                        "qs": qs, "gt": gt_answers,
                        "task_label": line["task_label"], 
                        "step_label": line["step_label"]}

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

        images = torch.from_numpy(frames).byte().to('cuda')
        images = images.permute(0, 3, 1, 2)

        qs = qs.replace("<video>\n", "")
        # convert qs to query, which contains only the question.
        query = qs.split("select one from options:")[0]
        possible_answers_str = "[" + ", ".join(list(line["index2ans"].values())) + "]"

        opts = ""
        for ii, one_opt in enumerate(list(line["index2ans"].values())):
            opts += ("({}) {}\n".format(ii+1, one_opt))
        possible_answers = opts.rstrip("\n")

        logger.info(f"{qid}\nquestion:{qs}\nanswer:{gt_answers}")
        total += 1

        try:
            code = get_code_video(query, input_type="video", extra_context=possible_answers_str)
            logger.info(f"code:\n{code}")
            # code = "def execute_command(video, possible_answers, query):\n    video_segment = VideoSegment(video)\n    last_frame = video_segment.frame_from_index(-1)\n    last_caption = last_frame.simple_query('What is happening in the frame?')\n    next_step = last_frame.best_text_match(option_list=possible_answers)\n    info = {'Caption of last frame': last_caption, 'Next step suggestion': next_step}\n    answer = video_segment.select_answer(info, query, possible_answers)\n    return answer"
            code = code.replace("def execute_command(video, possible_answers, query):", "")
            res = run_program([code, i, images, possible_answers, query], queues_in, input_type_="video")
            outputs = res[0]
            # logger.info(f"executed code:\n{res[1]}")
        except:
            vipergpt_error_cnt += 1
            outputs = None
            print('vipergpt encountered error')

        logger.info(f"output:\n{outputs}\nvipergpt_error_cnt: {vipergpt_error_cnt}")
        outputs = str(outputs)

        answer_id = parse_choice(outputs, line["all_choices"], line["index2ans"])
        results[qid]["response"] = outputs
        results[qid]["parser"] = answer_id
        # print("qid {}:\n{}".format(qid, qs))
        # print("AI: {}\nParser: {}\nGT: {}\n".format(outputs, answer_id, gt_answers))

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

        # update results
        print("save to {}".format(args.answers_file))
        with open(args.answers_file, "w") as f:
            json.dump(results, f, indent=2)

        et = datetime.now()
        logger.info(f"Execution time: {et-st}, global_acc: {global_acc.get_accuracy()}, avg_acc: {avg_acc}")

    logger.info(f'vipergpt_error_cnt: {vipergpt_error_cnt}')
    print("Process Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="data/COIN/videos")
    parser.add_argument("--question-file", type=str, default="data/testing_vqa19_25oct_v2_rm45_1.json")
    parser.add_argument("--answers-file", type=str, default="results/answers_vipergpt_gpt4omini_rm45_1.json")
    parser.add_argument("--num_video_frames", type=int, default=8)
    args = parser.parse_args()
    main(args)
