# Íris - IA
Este é o repositório utilizado para o versionamento do código-fonte referente aos modelos IA e treinamentos dos mesmo da equipe Íris. O Íris é um projeto de extração de informações de dados não-estruturados resultados de interações entre clientes e suporte utilizando técnicas de Processamento de Linguagem Natural.


![Python](https://img.shields.io/badge/Python%20-%20%233776AB?style=for-the-badge&logo=python&logoColor=white)![Pandas](https://img.shields.io/badge/Pandas-%23150458?style=for-the-badge&logo=pandas)![Static Badge](https://img.shields.io/badge/Scikitlearn-%23F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)




## Execução

1. **Clone o repositório**  
    Execute o comando abaixo para clonar o repositório em sua máquina local:
    ```bash
    git clone https://github.com/equipe-iris/Iris-Pre-Processamento.git
    ```

2. **Acesse o diretório do projeto**  
    Navegue até o diretório clonado:
    ```bash
    cd Iris-Pre-Processamento
    ```

3. **Crie o arquivo .env e as variáveis de ambiente**  
    No diretório raiz do projeto
    ```bash
    echo.> .env
    ```
    Copie o conteudo do arquivo ".env.example" no arquivo ".env"
    ```bash
    PORT=5000
    ALLOWED_ORIGINS=http://localhost:
    CLASSIFY_RESULTS_URL=http://localhost:7000/tickets/classification-results
    ```


4. **Crie o ambiente virtual e intalar as dependências**  
    Crie o amabiente de virtual para execução do projeto:
    ```bash
    python -m venv <nome_do_ambiente>
    ```
    Instale as dependências necessárias utilizando o gerenciador de pacotes 'pip'
    ```bash
    pip install requirements.txt
    ```

5. **Execute o servidor**  
    Inicie o servidor de desenvolvimento com o comando:
    ```bash
    python run.py
    ```
