# QA_From_Doc

An RAG with Agent, addionally add FAQ & Normal generate part.

### Prerequisites

#### 1. Install with pip from source

```
pip install -r requirments.txt
```

### Services APIs, and Deployment & Testing

#### Services APIs

| API             | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| **doc_input** | POST request to upload the files|
| **doc_delete** | POST request to delete sepicific file from the DB |
| **empty_collection** | POST request to delete sepicific collection of the DB |
| **qa_from_doc** | POST request to answer the user question by retrieve the related fragment from the DB |

#### Deployment
##### Start command
python api_server.py --port <port_num> --config_path <config_path> 
eg: python api_server.py --port 3011 --config_path './conf/config_test_aramus_qa.ini'

## Repository Organization

### `/`

| Subfolder                                                    | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [log] | Log will be stored in the /log/<application_name> folder |
| [uploaded] | All uploaded files will be saved in /uploaded/<application_name> folder |
| [m-split] | Manualy split files should be saved in /m-split/<application_name> folder and ADD script at document_embedding.py.  |

### Acknowledgements

* Mingsheng was responsible for the all qa_from_doc backend service include RAG, FAQ etc.
* Jinyu was responsible for the frontend interface.
* Fenglin was responsible for the backend service include workflow, FAQ management system etc.

