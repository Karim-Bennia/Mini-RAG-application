from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import ElementMetadata
from models import ProcessingEnum
from unstructured.documents.elements import Table


class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)


    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]
    

    def get_file_content(self, file_id: str):

        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(
            self.project_path,
            file_id
        )

        if file_ext == ProcessingEnum.PDF.value:
            return partition_pdf(filename=file_path,
                                 strategy="fast",
                                 hi_res_model_name="yolox",
                                 infer_table_structure=True,
                                 extract_image_block_types=["Image"],
                                 extract_image_block_to_payload=True,   
)

        if file_ext == ProcessingEnum.EXCEL.value:
            return partition_xlsx(filename=file_path,
                                  include_metadata=True,
                                  include_header=True,
                                  infer_table_structure=True
                                  )
        
        return None

   # def get_file_content(self, file_id: str):

    #    loader = self.get_file_loader(file_id=file_id)
     #   return loader

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int=500, overlap_size: int=50):
        print("before title chunk ", set([str(type(el)) for el in file_content]))
        chunks = chunk_by_title(file_content, 
                                max_characters=chunk_size, 
                                overlap=overlap_size,
                                combine_text_under_n_chars=100,       
                                new_after_n_chars=1500,
                                )
        for i, chunk in enumerate(chunks[50:100]):  
            print("\n=== Chunk", i, "===")
            print("Type:", type(chunk))
            print("Text:", chunk.text[:100], "...")  
            print("Metadata:", chunk.metadata)

        return chunks



    