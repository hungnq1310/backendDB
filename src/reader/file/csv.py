from pathlib import Path
from typing import List
from src.reader.base_reader import BaseReader
from src.node.base_node import TextNode
import csv

class CsvReader(BaseReader):
    """ Txt parser """

    # TODO: return is List
    def load_data(self, file: Path, extra_info=None) -> List[TextNode]:
        list_text = []
        # load_data returns a list of Document objects
        with open(file) as csv_file:
            data_lines = csv.reader(csv_file, delimiter=',')
            for idx, line in enumerate(data_lines):
                if idx == 0:
                    continue
                #update metadata
                metadata = {"file_name": file.name, "index": idx}
                if extra_info is not None:
                    metadata.update(extra_info)
                # append to list_text
                list_text.append(TextNode(
                    text=line[1],
                    extra_info=metadata
                ))
        
        return list_text



    