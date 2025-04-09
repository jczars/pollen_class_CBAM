import gdown

# ID do arquivo no Google Drive
file_id = "1h4-WdPeCxCTZ45mLxDCesyDjn8VMYv5A"

# URL formatada para download direto
url = f"https://drive.google.com/uc?id={file_id}"

# Nome do arquivo de saída
output_file = "Proventos_Pagos-2024-16568244.pdf"

# Baixar o arquivo
gdown.download(url, output_file, quiet=False)

print(f"Arquivo baixado com sucesso: {output_file}")