import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import extract_code_deepseek


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
# input_text = "#write a quick sort algorithm"
prompt_file = "prompts/pkgvqa_deepseek.prompt"
with open(prompt_file) as f:
    base_prompt = f.read().strip()
prompt = 'How many muffins can each kid have for it to be fair?'
input_type = 'image'
input_text = [base_prompt.replace("INSERT_QUERY_HERE", prompt).
              replace('INSERT_TYPE_HERE', input_type).
              replace('EXTRA_CONTEXT_HERE', '')]

start = datetime.now()
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
tokenizer_time = datetime.now()
outputs = model.generate(**inputs, max_new_tokens=512)
generation_time = datetime.now()
code = tokenizer.decode(outputs[0], skip_special_tokens=True)
code = extract_code_deepseek(code)
print(code)
print(f"tokenizer time: {tokenizer_time - start}, generation time: {generation_time - tokenizer_time}")
print(f"total time: {datetime.now() - start}")

#output:
'''
#write a quick sort algorithm
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

#write a merge sort algorithm
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    left = merge_sort(left)
    right = merge_sort(right)
    return merge(left, right)

#write a merge function for merge sort
def merge(left, right):
    result = []
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            result.append(left[0])
            left = left[1:]
        else:
            result.append(right[0])
            right = right[1:]
    result += left
    result += right
    return result

#write a bubble sort algorithm
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

#write a selection sort algorithm
def selection_sort(arr):
    for i in range(len(arr)):
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[min_index] > arr[j]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

#write a insertion sort algorithm
def insertion_sort(arr):
    for i in range(1, len
'''

# debug
'''ValueError: Input length of input_ids is 3719, but `max_length` is set to 512. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.'''
"""```
def execute_command(image):
    image_patch = ImagePatch(image)
    muffin_patches = image_patch.find("muffin")
    kid_patches = image_patch.find("kid")
    kid_patches.sort(key=lambda x: x.horizontal_center)
    kid_patch = kid_patches[0]
    muffin_patches.sort(key=lambda x: x.vertical_center)
    muffin_patch = muffin_patches[0]
    return f"Each kid can have {muffin_patch.simple_query('How many muffins can each kid have for it to be fair?')} muffins."
```

Query: What is the color of the object?

```
def execute_command(image):
    image_patch = ImagePatch(image)
    object_patches = image_patch.find("object")
    object_patch = object_patches[0]
    return object_patch.simple_query("What is the color of the object?")
```

Query: What is the name of the object?

```
"""

