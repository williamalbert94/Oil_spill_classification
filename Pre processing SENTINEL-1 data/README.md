# Pre-proccesing SENTINEL-1 Data

# Apply Orbit File


Durante o caminho do satélite, ele percorrerá uma rota relacionada ao parâmetro orbit. Isso permitirá que os dados sejam localizados usando informações de órbita, velocidade, latitude e longitude.

# Thermal Noise Removal


A remoção térmica de ruído reduz os efeitos de ruído na textura entre sub-faixas, em particular, normalizando o sinal de retroespalhamento em toda a cena do Sentinel-1 e resultando em descontinuidades reduzidas entre sub-faixas para cenas nos modos de aquisição de várias faixas. Sentinel-1 é perturbada pelo ruído térmico aditivo, particularmente no canal de polarização cruzada.

# Border Noise Removal


Ao gerar produtos de nível 1, é necessário corrigir o horário de início da amostragem para compensar a alteração da curvatura da Terra. Ao mesmo tempo, a compactação de azimute e alcance leva a artefatos radiométricos nas bordas da imagem. Remove ruídos de baixa intensidade e dados inválidos nas bordas da cena.

# Calibration


Calibração é o procedimento que converte valores de pixels digitais em retrodispersão SAR calibrada radiometricamente. A calibração reverte o fator de escala aplicado durante a geração do produto de nível 1 e aplica um deslocamento constante e um ganho dependente da faixa, incluindo a constante de calibração absoluta.

# Ellipsoid Correction RD

Dependendo do modelo utilizado no que diz respeito à representação do campo eletromagnético ao fazer a abertura do radar, será definida uma correção baseada no modelo de elipse. Isso dependerá das características dos dados do estudo. Isso retornará uma representação baseada no efeito Doppler, retornando uma representação física dos dados capturados.

# Conversion to dB

O coeficiente de retroespalhamento sem unidade é convertido em dB usando uma transformação logarítmica. Será uma representação virtual em db do retroespalhamento do pulso emitido ao fazer a abertura do radar e interagir com o objeto de interesse.


