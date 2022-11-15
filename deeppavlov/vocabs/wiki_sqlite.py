# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from logging import getLogger
from pathlib import Path
import pickle
from typing import List, Any, Optional, Union, Iterator

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator

logger = getLogger(__name__)


@register('wiki_sqlite_vocab')
class WikiSQLiteVocab(SQLiteDataIterator, Component):
    """Get content from SQLite database by document ids.

    Args:
        load_path: a path to local DB file
        join_docs: whether to join extracted docs with ' ' or not
        shuffle: whether to shuffle data or not

    Attributes:
        join_docs: whether to join extracted docs with ' ' or not

    """

    def __init__(self, load_path: str, join_docs: bool = True, shuffle: bool = False, **kwargs) -> None:
        SQLiteDataIterator.__init__(self, load_path=load_path, shuffle=shuffle)
        self.join_docs = join_docs

    def __call__(self, doc_ids: Optional[List[List[Any]]] = None, *args, **kwargs) -> List[Union[str, List[str]]]:
        """Get the contents of files, stacked by space or as they are.

        Args:
            doc_ids: a batch of lists of ids to get contents for

        Returns:
            a list of contents / list of lists of contents
        """
        all_contents = []
        if not doc_ids:
            logger.warning('No doc_ids are provided in WikiSqliteVocab, return all docs')
            doc_ids = [self.get_doc_ids()]

        for ids in doc_ids:
            contents = [self.get_doc_content(doc_id) for doc_id in ids]
            if self.join_docs:
                contents = ' '.join(contents)
            all_contents.append(contents)

        return all_contents



@register('wiki_sqlite_vocab_ru')
class WikiSQLiteVocabRu(Component):
    """Get content from Russian wikipedia dataset by document ids.

    Args:
        load_path: a path to local folder
    """

    def __init__(self, load_path: str, **kwargs) -> None:
        self.passages = PassageDB(load_path)

    def __call__(self, doc_ids: Optional[List[List[Any]]] = None) -> List[Union[str, List[str]]]:
        """Get the contents of files, stacked by space or as they are.

        Args:
            doc_ids: a batch of lists of ids to get contents for

        Returns:
            a list of contents / list of lists of contents
        """

        if not doc_ids:
            logger.warning('No doc_ids are provided, return an empty list')
            all_contents = []
        else:
            all_contents = [[] for _ in range(len(doc_ids))]
            for num, top_ids in enumerate(doc_ids):
                for idx in top_ids:
                    all_contents[num].append(self.passages[idx].text)
        
        return all_contents


@dataclasses.dataclass
class Passage:
    id: int
    title: str
    text: str


class PassageDB:
    def __init__(self, input_path: Path):
        self.files_paths = list(Path(input_path).iterdir())
        self._db = []
        for filepath in self.files_paths:
            if str(filepath).endswith("pickle"):
                with open(filepath, "rb") as psg:
                    self._db.extend(pickle.load(psg))

    def __reduce__(self):
        return (self.__class__, (self._input_file,))

    def __len__(self):
        return len(self._db)

    def __getitem__(self, id_: int) -> Passage:
        title, text = self._db[id_][1], self._db[id_][2]
        return Passage(id_, title, text)

    def __iter__(self) -> Iterator[Passage]:
        for id_num, psg in enumerate(self._db):
            title, text = psg[1], psg[2]
            yield Passage(id_num, title, text)