from datasets import GeneratorBasedBuilder
from datasets import DatasetInfo, Features, Value, ClassLabel
from datasets import DownloadManager, SplitGenerator, Split


class TUSA(GeneratorBasedBuilder):
    def _info(self):
        return DatasetInfo(
            features=Features({
                "text": Value("string"),
                "label": ClassLabel(names=["Negative", "Positive"])
            })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepath": "../dataset/tusa/train.txt"
                }
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "filepath": "../dataset/tusa/validation.txt"
                }
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "filepath": "../dataset/tusa/test.txt"
                }
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as file:
            for index, line in enumerate(file):
                yield index, {
                    "text": line.strip().split('\t')[1],
                    "label": line.strip().split('\t')[0]
                }
