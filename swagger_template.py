template = {
  "swagger": "2.0",
  "info": {
    "title": "QA from DOC API",
    "description": "QA from DOC",
    "version": "0.0.1"
  },
  "tags": [
    {
      "name": "qa_from_doc",
      "description": "qa_from_doc"
    }
  ],
  "paths": {
    "/doc_input": {
      "post": {
        "tags": [
          "doc_input"
        ],
        "summary": "Input doc with form text",
        "description": "",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "text",
            "required": True,
            "schema": {
              "$ref": "#/definitions/text"
            }
          },
        ],
        "responses": {
          "200": {
            "description": "Input status",
            "schema": {
              "$ref": "#/definitions/input_text"
            }
          }
        }
      }
    },
    "/qa_search": {
      "post": {
        "tags": [
          "qa_search"
        ],
        "summary": "Find the most similar QA pairs",
        "description": "",
        "consumes": [
          "multipart/form-data"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "name": "token_name",
            "in": "formData",
            "description": "User questions",
            "required": True,
            "type": "list"
          },
          {
            "name": "token_name",
            "in": "formData",
            "description": "Specific token name for the excel or the task",
            "required": True,
            "type": "string"
          }
        ],
        "responses": {
          "200": {
            "description": "Successful operation",
            "schema": {
              "$ref": "#/definitions/ApiResponse"
            }
          }
        }
      }
    },
    "/qa_from_doc": {
      "post": {
        "tags": [
          "qa_from_doc"
        ],
        "summary": "Answer the questions from doc",
        "description": "",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "question",
            "required": True,
            "schema": {
              "$ref": "#/definitions/question"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Answer",
            "schema": {
              "$ref": "#/definitions/answer"
            }
          }
        }
      }
    },
    "/empty_collection": {
      "post": {
        "tags": [
          "empty_collection"
        ],
        "summary": "Empty selected collections or all collections",
        "description": "",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "Given token name or leave empty will empty all collections",
            "required": True,
            "schema": {
              "type": "array",
              "items": {
                "$ref": "#/definitions/collection_name"
              }
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful operation",
            "schema": {
              "$ref": "#/definitions/ApiResponse"
            }
          }
        }
      }
    }
  },
  "definitions": {
    "text": {
      "type": "object",
      "required": [
        "text"
      ],
      "properties": {
        "text": {
          "type": "string",
          "items": {
            "type": "string"
          },
          "example": "Visual pollution is ......."
        },
        "filename": {
          "items": {
            "type": "string"
          },
          "example": "Qh2Xhknhgi8"
        },
      }
    },
    "input_text": {
      "type": "object",
      "properties": {
        "response": {
          "items": {
            "type": "string"
          },
          "example": "Save success!"
        },
        "status": {
          "items": {
            "type": "string"
          },
          "example": "Success!"
        },
        "running_time": {
          "items": {
            "type": "number"
          },
          "example": "0.0325"
        }
      }
    },
    "question": {
      "type": "object",
      "properties": {
        "question": {
          "items": {
            "type": "string"
          },
          "example": "What is Visual pollution?"
        },
        "filename": {
          "items": {
            "type": "string"
          },
          "example": "Qh2Xhknhgi8"
        },
        "messages": {
          "items": {
            "type": "list"
          },
          "example": []
        },
      }
    },
    "answer": {
      "type": "object",
      "properties": {
        "response": {
          "items": {
            "type": "string"
          },
          "example": "Visual pollution is ......"
        },
        "fragment": {
          "items": {
            "type": "string"
          },
          "example": "Doc: Visual pollution is ......"
        },
        "score": {
          "items": {
            "type": "float"
          },
          "example": 0.897364564
        },
        "document_name": {
          "items": {
            "type": "string"
          },
          "example": "xxx.txt"
        },
        "status": {
          "items": {
            "type": "string"
          },
          "example": "Success!"
        },
        "running_time": {
          "items": {
            "type": "number"
          },
          "example": "0.0325"
        }
      }
    },
    "ApiResponse": {
      "type": "object",
      "properties": {
        "code": {
          "type": "integer",
          "format": "int32"
        },
        "type": {
          "type": "string"
        },
        "message": {
          "type": "string"
        }
      }
    },
    "collection_name": {
      "type": "string",
      "example": "test"
    },
  }
}