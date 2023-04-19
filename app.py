from flask import Flask, render_template
from flask import request
import torch
import torchvision.transforms as transforms
import torch.nn.init
from PIL import Image

# torch.nn.Module:  PyTorch의 모든 Neural Network의 Base Class
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out



app = Flask(__name__)

@app.route("/")
def index() :

    return render_template("index.html")

@app.route('/uploader', methods=['POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']   # index.html에서 file이라는 이름의 요소가 가지고 있는 8.png를
                                    # 객체로 호출함
        print(f.filename)           # f 객체에서 filename을 가져옴. 8.png로 출력
        f.save("./images/" + f.filename) # f 를 현재 작업디렉토리 images 폴더에 8.png로 저장

        # 모델 interence 함수 호출
        output = infer("./images/" + f.filename)

        return render_template("CNN_result.html", result = output)

    #업로드 테스트
    #return '업로드 완료'
    
def infer(filename) :
    # 다시불러와서 추론 해보기
    model = CNN()
    model.load_state_dict(torch.load("cnn_model.pt", map_location=torch.device('cpu'))) 
    model.eval() #평가 모드로 설정하여야 합니다. 이 과정을 거치지 않으면 일관성 없는 추론 결과가 출력
    with torch.no_grad():

        # 이미지 파일 경로 설정
        img = Image.open(filename) # 이미지 열기

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # RGB(3D) -> Gray(2D)
            transforms.Resize((28, 28)), # 모델 인풋에 맞게
            transforms.ToTensor(), # 토치 텐서 타입으로 맞춰줘야함
        ])
        
        img_tensor = transform(img) # [1, 28, 28]
        # 모델이 원래 [배치사이즈, 채널, 가로, 세로] 이기때문에 [0]번째에 배치사이즈 추가
        img_tensor = img_tensor.unsqueeze(0) # [1, 1, 28, 28] 

        prediction = model(img_tensor)
                            
        result = torch.argmax(prediction, 1).tolist()[0]

    return result

if __name__ == "__main__" :
    app.run(debug=True)