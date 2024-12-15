from tqdm import tqdm
from colorama import Fore, Style
from colorama import init
init(autoreset=True)

class Bar():
    def __init__(self, iterable, desc="", color=Fore.LIGHTCYAN_EX):
        self.iterable = iterable
        self.desc = desc
        self.color = color
    
    def display(self):
        return tqdm(self.iterable, desc=f"{self.desc}", bar_format="{l_bar}%s{bar}%s{r_bar}" % (self.color, Style.RESET_ALL))