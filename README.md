# qa_from_doc

# Start command
python api_server.py --port <port_num> --config_path <config_path> 
eg: python api_server.py --port 3011 --config_path './conf/config_test_aramus_qa.ini'

Log will be stored in the /log/<application_name> folder

All uploaded files will be saved in /uploaded/<application_name> folder

Manualy split files should be saved in /m-split/<application_name> folder and ADD script at document_embedding.py. (The code start from line 84) 

