import os
from typing import Any, Dict, List, Optional, Tuple

from langchain.vectorstores.chroma import Chroma
from langchain.docstore.document import Document
from scenario.envScenario import EnvScenario

from LLMDriver.llm_backend import get_embedding_function


class DrivingMemory:
    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type
        if encode_type != 'sce_language':
            raise ValueError("Unknown ENCODE_TYPE: should be `sce_language`")

        self.embedding = get_embedding_function()
        db_path = os.path.join(
            './db', 'chroma_5_shot_20_mem/') if db_path is None else db_path
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )
        print(self.scenario_memory)
        print("==========Loaded Memory, Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.==========")

    def retriveMemory(self, driving_scenario: EnvScenario, controlled_vehicle_id: int, top_k: int = 5):
        """Baseline similarity-only retrieval (KoMA)."""
        query_scenario = driving_scenario.describe(controlled_vehicle_id)
        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=top_k)
        fewshot_results = []
        for idx in range(0, len(similarity_results)):
            fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    def retrieveMemoryWithScores(
        self,
        driving_scenario: EnvScenario,
        controlled_vehicle_id: int,
        top_k: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Return (metadata, distance, page_content) for verification filtering.
        Distance is chroma score (lower is closer for L2).
        """
        query_scenario = query_text or driving_scenario.describe(controlled_vehicle_id)
        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=top_k)
        out: List[Tuple[Dict[str, Any], float, str]] = []
        for doc, score in similarity_results:
            meta = dict(doc.metadata or {})
            # Ensure description available for factual/contextual checks
            if "sce_description" not in meta:
                meta["sce_description"] = doc.page_content
            meta["page_content"] = doc.page_content
            out.append((meta, float(score), doc.page_content))
        return out

    def KoMAaddMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None, comments: str = ""):
        sce_descrip = sce_descrip.replace("'", '')
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": sce_descrip
            }
        )

        if len(get_results['ids']) > 0:
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas={
                    "human_question": human_question,
                    'LLM_response': response,
                    'action': action,
                    'comments': comments
                }
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=sce_descrip,
                metadata={
                    "human_question": human_question,
                    'LLM_response': response, 'action': action,
                    'comments': comments
                }
            )
            id = self.scenario_memory.add_documents([doc])
            print("Add a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def addMemory(self, sce_descrip: str, human_question: str, response: str, plan: str, action: int, sce: EnvScenario = None, comments: str = ""):
        sce_descrip = sce_descrip.replace("'", '')
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": sce_descrip
            }
        )

        if len(get_results['ids']) > 0:
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas={
                    "human_question": human_question,
                    'LLM_response': response,
                    'plan': plan,
                    'action': action,
                    'comments': comments
                }
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=sce_descrip,
                metadata={
                    "human_question": human_question,
                    'LLM_response': response, 'plan': plan, 'action': action,
                    'comments': comments
                }
            )
            id = self.scenario_memory.add_documents([doc])
            print("Add a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)
        print("Delete", len(ids), "memory items. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        for i in range(0, len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] in current_documents['embeddings']:
                print("Already have one memory item, skip.")
            else:
                self.scenario_memory._collection.add(
                    embeddings=other_documents['embeddings'][i],
                    metadatas=other_documents['metadatas'][i],
                    documents=other_documents['documents'][i],
                    ids=other_documents['ids'][i]
                )
        print("Merge complete. Now the database has ", len(
            self.scenario_memory._collection.get(
                include=['embeddings'])['embeddings']), " items.")


if __name__ == "__main__":
    pass
