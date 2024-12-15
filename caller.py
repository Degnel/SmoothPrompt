import json
import os
from dotenv import load_dotenv
import openai
import anthropic
import base64
import tiktoken
import asyncio
from itertools import islice
from loading_animation.loading_bar import Bar
from colorama import Fore
import logging

class Call():
    def __init__(self, 
                 prompt_name,
                 model="gpt-4o-mini", 
                 temperature=0, 
                 max_tokens=None,
                 env_path=None, 
                 prompts_path=None, 
                 models_path=None, 
                 display=False,
                 max_retry=3,
                 time_sleep=1,
                 mini_batch_count=1, # 0 prend le temps maximal en envoyant les calls les uns après les autres
                 bar=True,
                 desc="",
                 color="GREEN",
                 hide_asyncio_errors=True,
                 **kwargs):
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.models_path = models_path
        self.display = display
        self.max_retry = max_retry
        self.time_sleep = time_sleep
        self.mini_batch_count = mini_batch_count
        self.bar = bar
        self.desc = desc
        self.color = getattr(Fore, color, None)
        self.hide_asyncio_errors = hide_asyncio_errors
        
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        
        if env_path:
            self.env_path = env_path
        else:
            self.env_path = os.path.join(absolute_path, '..', '.env')
        
        if not prompts_path:
            prompts_path = os.path.join(absolute_path, 'prompts.json')

        with open(prompts_path, 'r') as f:
            self.prompts = json.load(f)

        self.prompt_placeholder = self.prompts[prompt_name]

        if isinstance(self.prompt_placeholder, str):
            self.prompt_placeholder = [self.prompt_placeholder]

        if not models_path:
            models_path = os.path.join(absolute_path, 'models.json')

        with open(models_path, 'r') as f:
            self.models = json.load(f)
        
        self.harmonize_kwargs(**kwargs)

        self.set_images()

        if self.model in self.models["OpenAI"]:
            self.enc = tiktoken.encoding_for_model(self.model)

    def harmonize_kwargs(self, **kwargs):
        """Harmonize les kwargs pour qu'ils soient tous de la même longueur."""
        if kwargs:
            list_values = [value for value in kwargs.values() if isinstance(value, list)]
            
            if list_values:
                max_length = max(len(value) for value in list_values)
                
                for value in list_values:
                    if len(value) != max_length:
                        raise ValueError("Toutes les listes doivent avoir le même nombre d'éléments.")

                for key, value in kwargs.items():
                    if not isinstance(value, list):
                        kwargs[key] = [value] * max_length
                
                self.batch = True
                length = len(next(iter(kwargs.values())))
                self.kwargs = [{key: kwargs[key][i] for key in kwargs if not key.endswith('_img')} for i in range(length)]
                self.image_kwargs = [{key: kwargs[key][i] for key in kwargs if key.endswith('_img')} for i in range(length)]
            else:
                self.batch = False
                self.kwargs = [{key: value for key, value in kwargs.items() if not key.endswith('_img')}]
                self.image_kwargs = [{key: value for key, value in kwargs.items() if key.endswith('_img')}]
        else:
            self.batch = False
            self.kwargs = [{}]
            self.image_kwargs = [{}]

    def set_images(self):
        call_count = len(self.image_kwargs)
        message_count = len(self.prompt_placeholder)
        self.images = [[None]*message_count for _ in range(call_count)]

        for key in self.image_kwargs[0]:
            for j in range(message_count):
                if f"{{{key}}}" in self.prompt_placeholder[j]:
                    if sum(self.prompt_placeholder[j].count(f"{{{key}}}") for key in self.image_kwargs[0]) > 1:
                        raise ValueError("Il ne peut pas y avoir plusieurs images par message.")
                    for i in range(call_count):
                        self.images[i][j] = self.image_kwargs[i][key]

                    self.prompt_placeholder[j] = self.prompt_placeholder[j].replace(f"{{{key}}}", "").strip()

    def encode_image(self, image):
        """Encode une image en base64."""
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def run_command(self):
        load_dotenv(self.env_path)

        if self.model in self.models["OpenAI"]:
            self.api_type = "OpenAI"
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai.AsyncOpenAI(api_key=self.openai_api_key)

        elif self.model in self.models["Anthropic"]:
            self.api_type = "Anthropic"
            self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
            self.client = anthropic.AsyncAnthropic(api_key=self.claude_api_key)

        else:
            raise ValueError(f"Le modèle n'est pas présent dans le fichier situé à {self.models_path}.")
        
        answer = await self.batch_split()

        if self.display:
            print(answer)
            
        return answer
    
    async def batch_split(self):
        iterator = list(zip(self.kwargs, self.images))
        iterator = self.split_list(iterator, self.mini_batch_count)

        if self.bar:
            print()
            iterator = Bar(iterator, self.desc, color=self.color).display()

        responses = []
        for mini_batch in iterator:
            responses += await self.routine(mini_batch)
        
        if self.bar:
            print()
            
        if self.batch:
            return responses
        else:
            return responses[0]

    def split_list(self, lst, n):
        """Split une liste en n sous-listes."""
        if n == 0:
            return [[item] for item in lst]
    
        it = iter(lst)
        return [list(islice(it, i)) for i in [len(lst) // n + (1 if x < len(lst) % n else 0) for x in range(n)]]

    def exception_handler(self, loop, context):
        # print(context.keys())
        # print(type(context['future']), context['future'])
        pass
        # exception = context['exception']
        # message = context['message']
        # logging.error(f'Task failed, msg={message}, exception={exception}')

    async def routine(self, kwargs_images_list):
        tasks = []

        # loop = asyncio.get_running_loop()
        # if self.hide_asyncio_errors:
        #     loop.set_exception_handler(self.exception_handler)

        for kwargs, images in kwargs_images_list:
            prompt = self.parse_prompt(**kwargs)
            messages = self.create_messages(prompt, images)
            max_tokens = self.compute_max_tokens(prompt)
            task = asyncio.create_task(self.async_llm_call(messages, max_tokens))
            tasks.append(asyncio.shield(task))

        batch = await asyncio.gather(*tasks)

        return batch

    def parse_prompt(self, **kwargs):
        """Remplace les placeholders par les valeurs des kwargs."""
        # if isinstance(self.prompt_placeholder, str):
        #     prompt = self.prompt_placeholder.format(**kwargs)
        # elif isinstance(self.prompt_placeholder, list):
        try:
            prompt = [p.format(**kwargs) for p in self.prompt_placeholder]
        except:
            raise ValueError("Les paramètres mentionnés ne correspondent pas à ceux présents dans le prompt.")
        # else:
        #     raise ValueError("Les prompts doivent être une string ou une liste de string.")

        return prompt

    def create_messages(self, prompt, images):
        """Crée les messages à envoyer à l'API."""
        # Créer une conversation entre l'utilisateur et le LLM à partir d'une liste

        if isinstance(prompt, list):
            messages = []
            for i, message in enumerate(prompt):
                role = "user" if i % 2 == 0 else "assistant"
                if images[i]:
                    base64_image = self.encode_image(images[i])
                    content = [{"type": "text", "text": message},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
                else:
                    content = message
                
                messages.append({"role": role, "content": content})
        else:
            messages = [{"role": "user", "content": prompt}]
        
        return messages

    def compute_max_tokens(self, prompt):
        """Calcule le nombre de tokens à envoyer à l'API."""
        if isinstance(prompt, list):
            prompt = " ".join(prompt)

        if not self.max_tokens:
            if self.model in self.models["OpenAI"]:
                token_count = len(self.enc.encode(prompt))
                model = self.models["OpenAI"][self.model]
            else:
                token_count = anthropic.Client().count_tokens(prompt)
                model = self.models["Anthopic"][self.model]

            max_tokens = min(model["input"] - token_count, model["output"])
        else:
            max_tokens = self.max_tokens
        
        return max_tokens

    async def async_llm_call(self, messages, max_tokens):
        logging.info(f'Task is starting')
        for attempt in range(self.max_retry):
            try:
                if self.api_type == "OpenAI":
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=self.temperature
                    )
                    return response.choices[0].message.content
                
                elif self.api_type == "Anthropic":
                    message = await self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages
                    )
                    return message.content[0].text
                
            except Exception as e:
                print(f"Erreur lors de l'appel à l'API : {e}. Tentative {attempt + 1} sur {self.max_retry}.")
                if attempt < self.max_retry - 1:
                    await asyncio.sleep(1)
                else:
                    # raise Exception(f"L'appel à l'API a échoué après {self.max_retry} tentatives.")
                    return None
        logging.info(f'Task is done')
