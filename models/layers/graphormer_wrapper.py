import sys
import os


sys.path.append("/home/bitfra/Desktop/gatr_test/pixel2mesh_ga/MeshGraphormer/")
# print("Python Path:", sys.path)

# print("Current Working Directory:", os.getcwd())
# print("File Location:", __file__)


from src.modeling.bert.modeling_graphormer import Graphormer
from src.modeling.bert.e2e_body_network import Graphormer_Body_Network
from src.modeling.bert.e2e_body_network import Graphormer_Body_Network




prova = Graphormer_Body_Network()

# print(prova)