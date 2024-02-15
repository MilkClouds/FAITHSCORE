import argparse
import os
import pickle
import re
import time

import nltk
import numpy as np
import openai
from modelscope.pipelines import pipeline
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from modelscope.utils.constant import Tasks
from tqdm import tqdm

from faithscore.llama_pre import load_llama, stage1_llama
from faithscore.llava15 import LLaVA
from faithscore.utils import llava15, ofa

path = os.path.dirname(__file__)
cur_path = os.path.dirname(path)
cur_path = os.path.join(cur_path, "faithscore")


def filecache(name, fn, *args):
    path = f"cache/{name}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        result = fn(*args)
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result


class FaithScore:
    def __init__(self, vem_type, api_key=None, llava_path=None, tokenzier_path=None, use_llama=False, llama_path=None):
        openai.api_key = api_key
        max_seq_len = 500
        max_batch_size = 1
        self.use_llama = use_llama

        # self.vem_path = model_path
        self.model_type = vem_type  ### [ofa_ve, ofa, mplug, blip2, llava]
        model_list = ["ofa_ve", "ofa", "mplug", "blip2", "llava"]
        if vem_type not in model_list:
            print(f"Error: the model type {vem_type} not in {str(model_list)}")
            exit()
        self.llava_path = llava_path

        if use_llama:
            if llava_path and tokenzier_path:
                self.llama, self.tokenizer = load_llama(llama_path)
            else:
                print(f"Error: please input the model path for llama")
                exit()

    def call_openai(self, pts):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": pts},
                    ],
                    temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print(e)
                print("Continue......")
                time.sleep(10)

    async def async_call_openai_gpt4(self, pts):
        # be aware of cost! gpt-4-turbo is 200x more expensive than gpt-3.5-turbo
        while True:
            try:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "Strictly follow the user's instructions and strictly follow the example. Strictly follow the example form and do not say any characters that are not included in the form.",
                        },
                        {"role": "user", "content": pts},
                    ],
                    temperature=0.0,  # TODO: figure out which temperature is best for evaluation
                    timeout=2,
                )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print(e)
                print("Continue......")
                time.sleep(10)

    async def async_call_openai(self, pts):
        while True:
            try:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": pts},
                    ],
                    temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                    timeout=2,
                )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print(e)
                print("Continue......")
                time.sleep(10)

    def batched_call_openai(self, inputs):
        import asyncio
        import time

        from tqdm.auto import tqdm, trange

        # import multiprocessing
        # import joblib
        # from pqdm.processes import pqdm
        # results = joblib.Parallel(n_jobs=16)(joblib.delayed(self.call_openai)(pts) for pts in inputs)

        results = []
        api_batch_size = 11
        for batch in trange(0, len(inputs), api_batch_size):

            async def task():
                return await asyncio.gather(
                    *[self.async_call_openai(pts) for pts in inputs[batch : min(batch + api_batch_size, len(inputs))]]
                )

            results = results + asyncio.run(task())
            time.sleep(0.8)

        # print(inputs, len(inputs))
        # results = pqdm(inputs, self.call_openai, n_jobs=16, exception_behaviour="immediate")
        return results

    def stage1(self, answers):
        with open(os.path.join(cur_path, "prompts/prompt_label_des_ana.txt"), "r") as f:
            prompt_label_des_ana = f.read() + "\n\n"
        tasks = []
        for id in range(len(answers)):
            if not self.use_llama:
                pts = prompt_label_des_ana + answers[id].replace("\n", " ") + "\n" + "Labeled text: "
                tasks.append(pts)
        outputs = self.batched_call_openai(tasks)
        des_ana = []
        for id in range(len(answers)):
            if not self.use_llama:
                pts = prompt_label_des_ana + answers[id].replace("\n", " ") + "\n" + "Labeled text: "
                # des_ana.append(self.call_openai(pts).replace("\n", ""))
                des_ana.append(outputs[id].replace("\n", ""))
            else:
                pts = stage1_llama(self.llama, self.tokenizer, answers[id].replace("\n", " "))
                # print(pts)
                des_ana.append(pts)
            # exit()
        return des_ana

    def stage2(
        self,
        labeld_sub_sen,
    ):
        all_texts = []
        for ss in labeld_sub_sen:
            desc = ""
            pos_des = [substr.start() for substr in re.finditer("[D]", ss)]
            pos_ana = [substr.start() for substr in re.finditer("[A]", ss)]
            pos_seg = pos_des + pos_ana
            pos_seg.sort()
            for i in range(len(pos_seg)):
                if pos_seg[i] in pos_des:
                    if i == 0:
                        desc += ss[: pos_seg[i] - 1]
                    else:
                        desc += ss[pos_seg[i - 1] + 3 : pos_seg[i] - 1]
            all_texts.append(desc.replace("\n", " "))

        with open(os.path.join(cur_path, "prompts/prompt_de_atomic.txt"), "r") as f:
            prompt_de_atomic = f.read()
        Entities = []
        Relations = []
        Colors = []
        Counting = []
        Others = []

        results = []
        nons = "Entities:\nRelations:\nColors:\nCounting:\nOther attributes:"
        futures = []
        for ans in all_texts:
            ans = ans.replace("\n", "")
            pts = prompt_de_atomic + "\nAnswer: " + ans
            futures.append(pts)
            # response = self.call_openai(pts)
        outputs = self.batched_call_openai(futures)
        index = 0
        for ans in all_texts:
            ans = ans.replace("\n", "")
            pts = prompt_de_atomic + "\nAnswer: " + ans
            if ans == "":
                results.append(nons)
                continue
            # response = self.call_openai(pts)
            response = outputs[index]
            index += 1
            if "Entities" in response:
                results.append(response)
            else:
                results.append(nons)

        for facts in results:
            lines = facts.split("\n")
            for line in lines:
                if line[:9] == "Entities:":
                    entity = line.strip().replace("Entities: ", "").split(". ")
                    if line.strip() == "Entities:":
                        entity = []
                    Entities.append(entity)
                if line[:10] == "Relations:":
                    # print(line.strip().replace("Relations: ","").replace("],","]],").split("], "))
                    relation = line.strip().replace("Relations: ", "").split(". ")
                    if line.strip() == "Relations:":
                        relation = []
                    Relations.append(relation)
                if line[:7] == "Colors:":
                    color = line.strip().replace("Colors: ", "").split(". ")
                    if line.strip() == "Colors:":
                        color = []
                    Colors.append(color)
                if line[:9] == "Counting:":
                    count = line.strip().replace("Counting: ", "").split(". ")
                    if line.strip() == "Counting:":
                        count = []
                    Counting.append(count)
                if line[:17] == "Other attributes:":
                    other = line.strip().replace("Other attributes: ", "").split(". ")
                    if line.strip() == "Other attributes:":
                        other = []
                    Others.append(other)

        hallucinations = [
            Entities[i] + Relations[i] + Colors[i] + Counting[i] + Others[i] for i in range(len(Entities))
        ]
        # print(hallucinations)
        return hallucinations, Entities, Relations, Colors, Counting, Others

    def stage3(self, atomic_facts, images, img_path=None):
        # ofa_pipe = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_large_en')
        # model = pipeline(Tasks.visual_entailment, model=self.vem_path)
        if getattr(self, "model", None) is None:
            if self.model_type == "ofa_ve":
                model = pipeline(Tasks.visual_entailment, model="damo/ofa_visual-entailment_snli-ve_large_en")

            if self.model_type == "ofa":
                preprocessor = OfaPreprocessor(model_dir="damo/ofa_visual-question-answering_pretrain_large_en")
                model = pipeline(
                    Tasks.visual_question_answering,
                    model="damo/ofa_visual-question-answering_pretrain_large_en",
                    model_revision="v1.0.1",
                    preprocessor=preprocessor,
                )

            if self.model_type == "llava":
                if not self.llava_path:
                    print("Please input path for LLaVA model.")
                    exit()
                model = LLaVA(model_path=self.llava_path)
            self.model = model
        model = self.model

        tasks = []
        for id, elements in enumerate(atomic_facts):
            if img_path:
                image = os.path.join(img_path, images[id])
            else:
                image = images[id]

            for element in elements:
                # input = {'image': image, 'text': element}
                prompt = (
                    "Statement: "
                    + element
                    + " Is this statement is right according to the image? Please answer yes or no."
                )
                tasks.append([element, image])
                # if self.model_type == "ofa_ve":
                #     output = ofa(True, model, element, image)
                # if self.model_type == "ofa":
                #     output = ofa(False, model, prompt, image)
                # if self.model_type == "llava":
                #     output = llava15(image, prompt, model)
        from modelscope.outputs import OutputKeys

        inputs = [{"image": image, "text": element} for element, image in tasks]
        print(len(inputs), len(atomic_facts))
        outputs = model(inputs, batch_size=16)
        # print(outputs)
        # list of {'labels': ['yes'], 'scores': [1.0], 'samples': {'image': '/home/claude/datasets/stanford_image_paragraph/stanford_img/content/stanford_images/2403904.jpg', 'text': 'The trees across from the stop sign are green and healthy'}}, {'labels': ['yes'], 'scores': [1.0], 'samples': {'image': '/home/claude/datasets/stanford_image_paragraph/stanford_img/content/stanford_images/2403904.jpg', 'text': 'The sky is a little cloudy.'}}
        outputs = [output[OutputKeys.LABELS][0] for output in outputs]

        global_index = 0
        fact_scores = []
        for id, elements in enumerate(atomic_facts):
            fact_score = []
            for element in elements:
                output = outputs[global_index]
                global_index += 1
                if "yes" in output.lower():
                    fact_score.append(1)
                else:
                    fact_score.append(0)

                # if output.lower() == "yes" or output== "Yes":
                #     fact_score.append(1)
                # else:
                #     fact_score.append(0)
            fact_scores.append(fact_score)
            # result.append(output[OutputKeys.LABELS])
            # results.append({"image": images_id[id], "facts": elements, "result": str(result)})
            # checking_results.append(result)

        instance_score = [sum(ii) / len(ii) if len(ii) > 0 else 0 for ii in fact_scores]
        # print("Overall score: ", sum(instance_score) / len(instance_score))

        return sum(instance_score) / len(instance_score), fact_scores

    """
    answers: a list of strings, each element in this list is an answer
    """

    def faithscore(self, answers, images):
        ## Stage 1: Sub-setence Identification
        labeld_sub_sen = self.stage1(answers)
        ### Stage 2: Atomic Fact Generation
        atomic_facts, Entities, Relations, Colors, Counting, Others = self.stage2(labeld_sub_sen)
        ### Stage 3: Verification
        # print(atomic_facts)
        score, fact_scores = self.stage3(atomic_facts, images)
        sentence_score = self.sentence_faithscore(
            Entities, Relations, Colors, Counting, Others, self.labeled_sub(labeld_sub_sen), fact_scores
        )
        return score, sentence_score

    def cached_faithscore(self, name, answers, images):
        # file-cache each step with name
        # if file exists, load and return
        # else, run faithscore and save to file

        labeld_sub_sen = filecache(f"{name}_stage1", self.stage1, answers)
        atomic_facts, Entities, Relations, Colors, Counting, Others = filecache(
            f"{name}_stage2", self.stage2, labeld_sub_sen
        )
        score, fact_scores = filecache(f"{name}_stage3", self.stage3, atomic_facts, images)
        score_per_type = filecache(
            f"{name}_score_per_type",
            self.get_score_per_type,
            fact_scores,
            Entities,
            Relations,
            Colors,
            Counting,
            Others,
        )
        sentence_score = filecache(
            f"{name}_sentence_faithscore",
            self.sentence_faithscore,
            Entities,
            Relations,
            Colors,
            Counting,
            Others,
            self.labeled_sub(labeld_sub_sen),
            fact_scores,
        )
        sentence_score_per_type = [
            self.sentence_faithscore(
                Entities, Relations, Colors, Counting, Others, self.labeled_sub(labeld_sub_sen), fact_scores, i
            )
            for i in range(5)
        ]

        # write score, score_per_type, sentence_score as txt file
        with open(f"cache/{name}.txt", "w") as f:
            f.write(f"score: {score}\n")
            f.write(f"score_per_type: {score_per_type}\n")
            f.write(f"sentence_score: {sentence_score}\n")
            f.write(f"sentence_score_per_type: {sentence_score_per_type}\n")
        return score, score_per_type, sentence_score, sentence_score_per_type

    def visualize_result(self, name, answers, images):
        labeld_sub_sen = filecache(f"{name}_stage1", self.stage1, answers)
        atomic_facts, Entities, Relations, Colors, Counting, Others = filecache(
            f"{name}_stage2", self.stage2, labeld_sub_sen
        )
        score, fact_scores = filecache(f"{name}_stage3", self.stage3, atomic_facts, images)
        # visualize 2 results in one image
        from pathlib import Path

        import matplotlib.pyplot as plt
        from PIL import Image

        Path("result").mkdir(exist_ok=True, parents=True)

        for i in range(8):
            image_path = images[i]
            # show image and entities, relations, colors, counting, others in one plot
            img = Image.open(image_path)
            plt.imshow(img)
            # insert text
            # text = ""
            # if len(Entities[i]) > 0:
            #     text += f"Entities: {Entities[i]}\n"
            # if len(Relations[i]) > 0:
            #     text += f"Relations: {Relations[i]}\n"
            # if len(Colors[i]) > 0:
            #     text += f"Colors: {Colors[i]}\n"
            # if len(Counting[i]) > 0:
            #     text += f"Counting: {Counting[i]}\n"
            # if len(Others[i]) > 0:
            #     text += f"Others: {Others[i]}\n"
            text = f"image:{Path(images[i]).name}\nanswer:{answers[i]}\nscores: {fact_scores[i]}\nEntities: {Entities[i]}\nRelations: {Relations[i]}\nColors: {Colors[i]}\nCounting: {Counting[i]}\nOthers: {Others[i]}"
            with open(f"result/{name}_stage2_{i}.txt", "w") as f:
                f.write(text)
            # plt.text(0, 0, text, fontsize=12, color="red")
            plt.savefig(f"result/{name}_stage2_{i}.png")
        return

    def get_score_per_type(self, fact_scores, entities, relations, colors, counting, others):
        scores = []
        entity_scores = []
        relation_scores = []
        color_scores = []
        count_scores = []
        other_scores = []
        for i in range(len(fact_scores)):
            scores.append(fact_scores[i])
            entity_scores.append(fact_scores[i][: len(entities[i])])
            relation_scores.append(fact_scores[i][len(entities[i]) : len(entities[i]) + len(relations[i])])
            color_scores.append(
                fact_scores[i][
                    len(entities[i]) + len(relations[i]) : len(entities[i]) + len(relations[i]) + len(colors[i])
                ]
            )
            count_scores.append(
                fact_scores[i][
                    len(entities[i])
                    + len(relations[i])
                    + len(colors[i]) : len(entities[i])
                    + len(relations[i])
                    + len(colors[i])
                    + len(counting[i])
                ]
            )
            other_scores.append(
                fact_scores[i][len(entities[i]) + len(relations[i]) + len(colors[i]) + len(counting[i]) :]
            )
        mean = lambda x: np.mean(x) if len(x) > 0 else 0
        wrong_sum = lambda x: np.sum(1 - np.array(x))
        scores = [(mean(x), wrong_sum(x)) for x in scores]
        entity_scores = [(mean(x), wrong_sum(x)) for x in entity_scores]
        relation_scores = [(mean(x), wrong_sum(x)) for x in relation_scores]
        color_scores = [(mean(x), wrong_sum(x)) for x in color_scores]
        count_scores = [(mean(x), wrong_sum(x)) for x in count_scores]
        other_scores = [(mean(x), wrong_sum(x)) for x in other_scores]
        scores = (mean([x[0] for x in scores]), mean([x[1] for x in scores]))
        entity_scores = (mean([x[0] for x in entity_scores]), mean([x[1] for x in entity_scores]))
        relation_scores = (mean([x[0] for x in relation_scores]), mean([x[1] for x in relation_scores]))
        color_scores = (mean([x[0] for x in color_scores]), mean([x[1] for x in color_scores]))
        count_scores = (mean([x[0] for x in count_scores]), mean([x[1] for x in count_scores]))
        other_scores = (mean([x[0] for x in other_scores]), mean([x[1] for x in other_scores]))
        return scores, entity_scores, relation_scores, color_scores, count_scores, other_scores

    def sentence_faithscore(
        self, Entities, Relations, Colors, Counting, Others, all_texts, fact_scores, use_only=None
    ):
        Entities_recog = []

        for ents in Entities:
            entities = []
            for ent in ents:
                ent4sen = []
                sentence = nltk.sent_tokenize(ent)
                tags = nltk.pos_tag(nltk.word_tokenize(sentence[0]))
                for tag in tags:
                    if tag[1] in ["NN", "NNS", "JJ", "NNP", "VBG", "JJR", "NNPS", "RB", "DT"]:
                        # print(tag)
                        ent4sen.append(tag[0])
                    # tags.append(chunk.label())
                if len(ent4sen) < 1:
                    print(tags)
                    ent4sen = ["None"]
                entities.append(ent4sen[-1])

            # print(ents)
            # print(entities)
            if len(entities) != len(ents):
                print("error")
                exit()
            Entities_recog.append(entities)

        entity_scores = []
        relation_scores = []
        color_scores = []
        count_scores = []
        other_scores = []

        for i in range(len(fact_scores)):
            entity_scores.append(fact_scores[i][: len(Entities[i])])
            relation_scores.append(fact_scores[i][len(Entities[i]) : len(Entities[i]) + len(Relations[i])])
            color_scores.append(
                fact_scores[i][
                    len(Entities[i]) + len(Relations[i]) : len(Entities[i]) + len(Relations[i]) + len(Colors[i])
                ]
            )
            count_scores.append(
                fact_scores[i][
                    len(Entities[i])
                    + len(Relations[i])
                    + len(Colors[i]) : len(Entities[i])
                    + len(Relations[i])
                    + len(Colors[i])
                    + len(Counting[i])
                ]
            )
            other_scores.append(
                fact_scores[i][len(Entities[i]) + len(Relations[i]) + len(Colors[i]) + len(Counting[i]) :]
            )

        sentence_scores = []
        for id1, ins in enumerate(all_texts):
            sentence_score = []
            for id2, sub_sen in enumerate(all_texts[id1]):
                flag = True
                # print(Entities_recog)
                # print(entity_scores)
                for id3, ee in enumerate(Entities_recog[id1]):
                    if ee in sub_sen and entity_scores[id1][id3] != 1 and use_only in (0, None):
                        flag = False
                    for id4, rel in enumerate(relation_scores[id1]):
                        if ee in sub_sen and ee in Relations[id1][id4] and rel != 1 and use_only in (1, None):
                            flag = False
                    for id4, rel in enumerate(color_scores[id1]):
                        if ee in sub_sen and ee in Colors[id1][id4] and rel != 1 and use_only in (2, None):
                            flag = False
                    for id4, rel in enumerate(count_scores[id1]):
                        if ee in sub_sen and ee in Counting[id1][id4] and rel != 1 and use_only in (3, None):
                            flag = False
                    for id4, rel in enumerate(other_scores[id1]):
                        if ee in sub_sen and ee in Others[id1][id4] and rel != 1 and use_only in (4, None):
                            flag = False

                sentence_score.append(flag)
            sentence_scores.append(sentence_score)

        score4sen = [sum(ss) / len(ss) if len(ss) > 0 else 1 for ss in sentence_scores]
        sentence_level_score = score4sen
        # print(score4sen)
        # print(sum(score4sen)/len(score4sen))
        return sum(score4sen) / len(score4sen)

    def labeled_sub(self, des_ana):
        all_texts = []
        for ss in des_ana:
            desc = []
            pos_des = [substr.start() for substr in re.finditer("[D]", ss)]
            pos_ana = [substr.start() for substr in re.finditer("[A]", ss)]
            pos_seg = pos_des + pos_ana
            pos_seg.sort()
            for i in range(len(pos_seg)):
                if pos_seg[i] in pos_des:
                    if i == 0:
                        desc.append(ss[: pos_seg[i] - 1])
                    else:
                        desc.append(ss[pos_seg[i - 1] + 3 : pos_seg[i] - 1])
            all_texts.append(desc)
        return all_texts
