from core.spacy_utils import rough_split, init_nlp
from core.utils.paths import _3_1_SPLIT_BY_NLP
from core.utils import check_file_exists

@check_file_exists(_3_1_SPLIT_BY_NLP)
def split_by_spacy():
    nlp = init_nlp()
    rough_split(nlp)
    
    # 将 rough_split 的结果复制为 _3_1_SPLIT_BY_NLP
    from core.spacy_utils.load_nlp_model import ROUGH_SPLIT_FILE
    import shutil
    shutil.copy(ROUGH_SPLIT_FILE, _3_1_SPLIT_BY_NLP)
    
    return

if __name__ == '__main__':
    split_by_spacy()
