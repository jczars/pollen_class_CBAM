import os
import zipfile
import shutil

# Diretórios principais
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Diretório raiz do projeto
DATABASE_DIR = os.path.join(PROJECT_DIR, "BD")  # Pasta BD dentro do projeto

# URL da base de dados POLLEN23E
DATABASE_URL = "https://figshare.com/ndownloader/articles/1525086/versions/1"

def ask_user(prompt):
    """
    Pergunta ao usuário se deseja prosseguir com uma ação.
    Retorna True se o usuário responder 's' (sim), False caso contrário.
    """
    response = input(f"{prompt} (s/n): ").strip().lower()
    return response == "s"

def download_database(url, output_file):
    """
    Faz o download do arquivo ZIP usando wget, se o usuário concordar.
    """
    if os.path.exists(output_file):
        print(f"O arquivo '{output_file}' já existe.")
        if not ask_user("Deseja baixar novamente?"):
            return

    try:
        import subprocess
        print(f"Baixando o arquivo '{output_file}'...")
        subprocess.run(["wget", "-O", output_file, url], check=True)
        print(f"Download concluído: '{output_file}'.")
    except Exception as e:
        print(f"Erro ao baixar o arquivo: {e}")
        exit(1)

def extract_and_rename_zip(zip_path, target_dir, new_name):
    """
    Extrai o conteúdo do arquivo ZIP para uma pasta temporária, renomeia a pasta e organiza as imagens,
    se o usuário concordar.
    """
    extracted_dir = os.path.join(target_dir, new_name)
    temp_dir = os.path.join(target_dir, "1525086")

    if os.path.exists(extracted_dir) or os.path.exists(temp_dir):
        print(f"A pasta '{new_name}' ou '{temp_dir}' já existe.")
        if not ask_user("Deseja extrair novamente?"):
            return extracted_dir if os.path.exists(extracted_dir) else temp_dir

    try:
        # Cria a pasta temporária
        os.makedirs(temp_dir, exist_ok=True)

        # Extrai o conteúdo do ZIP para a pasta temporária
        print(f"Extraindo o arquivo '{zip_path}' para '{temp_dir}'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"Extração concluída em '{temp_dir}'.")

        # Renomeia a pasta temporária para o novo nome
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)  # Remove a pasta se já existir
        os.rename(temp_dir, extracted_dir)
        print(f"Pasta renomeada para '{new_name}'.")

        return extracted_dir
    except Exception as e:
        print(f"Erro durante a extração ou renomeação: {e}")
        exit(1)

def organize_images_by_class(base_dir):
    """
    Organiza as imagens em subpastas com base nos nomes das classes extraídos dos nomes dos arquivos.
    """
    print(f"Organizando imagens em '{base_dir}' por classe...")
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)

        # Processa apenas arquivos (ignora pastas)
        if os.path.isfile(file_path):
            # Remove a extensão do arquivo
            class_name = os.path.splitext(filename)[0]

            # Remove números e caracteres especiais após o nome principal
            class_name = ''.join([char for char in class_name if not char.isdigit() and char != '_']).strip()

            # Remove parênteses e espaços extras
            class_name = class_name.split('(')[0].strip()

            if not class_name:  # Caso o nome da classe não seja válido
                print(f"Erro ao extrair o nome da classe do arquivo: {filename}")
                continue

            # Cria uma subpasta para a classe, se ainda não existir
            class_dir = os.path.join(base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Move o arquivo para a subpasta correspondente
            new_file_path = os.path.join(class_dir, filename)
            shutil.move(file_path, new_file_path)
            print(f"Arquivo '{filename}' movido para a pasta '{class_name}'.")

    print("Organização concluída.")

def main():
    # Define os caminhos para o arquivo ZIP e a pasta de destino
    os.makedirs(DATABASE_DIR, exist_ok=True)  # Garante que a pasta BD existe
    zip_file = os.path.join(DATABASE_DIR, "1525086.zip")
    renamed_dir_name = "POLLEN23E"

    # Passo 1: Baixa o arquivo ZIP
    download_database(DATABASE_URL, zip_file)

    # Passo 2: Extrai o ZIP e renomeia a pasta
    extracted_dir = extract_and_rename_zip(zip_file, DATABASE_DIR, renamed_dir_name)

    # Passo 3: Organiza as imagens em pastas de acordo com as classes
    organize_images_by_class(extracted_dir)

    print("Processamento concluído!")

if __name__ == "__main__":
    main()