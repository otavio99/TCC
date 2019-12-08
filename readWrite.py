import csv

def carregar_dados(path):
	arquivo = open(path, "r")
	leitor = csv.reader(arquivo, delimiter=';')
	dias= []
	for dia in leitor:
 
 		#dado = [dia]
		dias.append(dia)

	return dias

def escrever_dados(path, dados):
	with open('file2.csv', 'w', newline='') as csvfile:
		writer=csv.writer(csvfile, delimiter=';')
		for i in range(len(dados)):
			writer.writerow(dados[i])


dia= carregar_dados('file.csv')
escrever_dados('file2.csv', dia)

