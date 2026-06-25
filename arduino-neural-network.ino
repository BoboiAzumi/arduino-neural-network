const int wb_max_dim = 8;
const int max_topo = 48;

class Tensor {
  private:
    float _value;
    float _grad;
    char _op;
    Tensor* _left;
    Tensor* _right;
    bool _visited;
    bool _visitedDelete;
    bool _isLeaf;

  public:
    Tensor(float value){
      _value = value;
      _grad = 0;
      _op = 0;
      _left = nullptr;
      _right = nullptr; 
      _visited = false;
      _isLeaf = true;
      _visitedDelete = false;
    }

    float getValue(){
      return _value;
    }

    void setValue(float value){
      _value = value;
      return;
    }

    float getGrad(){
      return _grad;
    }

    void setGrad(float grad){
      _grad = grad;
      return;
    }

    char getOp(){
      return _op;
    }

    void setOp(char op){
      _op = op;
      return;
    }

    bool isLeaf(){
      return _isLeaf;
    }

    void setIsLeaf(bool isLeaf){
      _isLeaf = isLeaf;
    }

    void setVisitedDelete(bool visitedDelete){
      _visitedDelete = visitedDelete;
    }

    Tensor* add(Tensor* other){
      Tensor* result = new Tensor(_value + other->getValue());

      result->setOp('+');
      result->setIsLeaf(false);
      result->_left = this;
      result->_right = other;

      result->_left->setVisitedDelete(false);
      result->_right->setVisitedDelete(false);

      return result;
    }

    Tensor* mul(Tensor* other){
      Tensor* result = new Tensor(_value * other->getValue());

      result->setOp('*');
      result->setIsLeaf(false);
      result->_left = this;
      result->_right = other;
      result->_left->setVisitedDelete(false);
      result->_right->setVisitedDelete(false);

      return result;
    }

    Tensor* power(Tensor* other){
      Tensor* result = new Tensor(pow(_value, other->getValue()));

      result->setOp('^');
      result->setIsLeaf(false);
      result->_left = this;
      result->_right = other;
      result->_left->setVisitedDelete(false);
      result->_right->setVisitedDelete(false);

      return result;
    }

    void backwardNode(){
      if(_op == '+'){
        _left->setGrad(_left->getGrad() + _grad);
        _right->setGrad(_right->getGrad() + _grad);
      }

      if(_op == '*'){
        _left->setGrad(
          _left->getGrad() + (_right->getValue() * _grad)
        );
        _right->setGrad(
          _right->getGrad() + (_left->getValue() * _grad)
        );
      }

      if(_op == '^'){
        _left->setGrad(
          _left->getGrad() + ((
            _right->getValue() * pow(_left->getValue(), _right->getValue() - 1)
          ) * _grad)
        );
        _right->setGrad(
          _right->getGrad() + (pow(_left->getValue(), _right->getValue()) * log(_left->getValue()) * _grad)
        );
      }
    }

    void buildTopo(Tensor** topo, int& topoSize){
      if(_visited){
        return;
      }

      _visited = true;

      if(_left){
        _left->buildTopo(
          topo,
          topoSize
        );
      }

      if(_right){
        _right->buildTopo(
          topo,
          topoSize
        );
      }

      topo[topoSize++] = this;
    }

    void clearVisited(){
      _visited = false;

      if(_left)
        _left->clearVisited();

      if(_right)
        _right->clearVisited();
    }

    void backward(){
      Tensor* topo[max_topo];
      int topoSize = 0;

      buildTopo(
        topo,
        topoSize
      );

      _grad = 1.0;

      for(int i = topoSize - 1; i >= 0; i--){
        topo[i]->backwardNode();
      }

      clearVisited();
    }

    void freeGraph() {
      if (_visitedDelete) {
        return;
      }

      _visitedDelete = true;

      if (_left){
        _left->freeGraph();
      }

      if (_right) {
        _right->freeGraph();
      }

      if (!_isLeaf){
        delete this;
      }
    }

    void zeroGrad() {
      _grad = 0.0;
      if (_left) {
        _left->zeroGrad();
      }
      if (_right) {
        _right->zeroGrad();
      }
    }

    Tensor* mseLoss(Tensor* target) {
      Tensor* negOne = new Tensor(-1.0);
      negOne->setIsLeaf(false);

      Tensor* negTarget = negOne->mul(target);
      Tensor* diff = this->add(negTarget);

      Tensor* exponent = new Tensor(2.0);
      exponent->setIsLeaf(false);

      Tensor* sq = diff->power(exponent);
      return sq;
    }
};

class Layer{
  private:
    Tensor* _w[wb_max_dim][wb_max_dim];
    Tensor* _b[wb_max_dim];
    int _units;
    int _inputSize;
  
  public:
    Layer(int units, int inputSize){
      _units = units;
      _inputSize = inputSize;

      for(int i = 0; i < _units; i++){
        for(int j = 0; j < _inputSize; j++){
          _w[i][j] = new Tensor((float)random(1, 100) / 100.0);
        }
      }

      for(int i = 0; i < _units; i++){
        _b[i] = new Tensor((float)random(1, 100) / 100.0);
      }
    }

    int getUnits(){
      return _units;
    }

    Tensor** forward(Tensor* *input){
      Tensor* *result = new Tensor*[_units];

      for(int i = 0; i < _units; i++){
        Tensor* temp = new Tensor(0);
        temp->setIsLeaf(false);
        for(int j = 0; j < _inputSize; j++){
          temp = temp->add(input[j]->mul(_w[i][j]));
        }
        result[i] = temp;
      }

      for(int i = 0; i < _units; i++){
        result[i] = result[i]->add(_b[i]);
      }

      return result;
    };

    void updateWeights(float lr) {
      for (int i = 0; i < _units; i++) {
        for (int j = 0; j < _inputSize; j++) {
          float newVal = _w[i][j]->getValue() - lr * _w[i][j]->getGrad();
          _w[i][j]->setValue(newVal);
        }
        float newBias = _b[i]->getValue() - lr * _b[i]->getGrad();
        _b[i]->setValue(newBias);
      }
    }

    void zeroGradWeights() {
      for (int i = 0; i < _units; i++) {
        for (int j = 0; j < _inputSize; j++) {
          _w[i][j]->setGrad(0.0);
        }
        _b[i]->setGrad(0.0);
      }
    }
};

Tensor* xdata[4][2] = {
  {
    new Tensor(1.0), new Tensor(2.0)
  },
  {
    new Tensor(2.0), new Tensor(3.0)
  },
  {
    new Tensor(3.0), new Tensor(4.0)
  },
  {
    new Tensor(4.0), new Tensor(5.0)
  },
};

Tensor* ydata[4][1] = {
  {
    new Tensor(3.0)
  },
  {
    new Tensor(4.0)
  },
  {
    new Tensor(5.0)
  },
  {
    new Tensor(6.0)
  },
};

Tensor* xdatatest[4][2] = {
  {
    new Tensor(1.5), new Tensor(2.5)
  },
  {
    new Tensor(10.0), new Tensor(11.0)
  },
  {
    new Tensor(8.0), new Tensor(9.0)
  },
  {
    new Tensor(7.5), new Tensor(8.5)
  },
};

Tensor* ydatatest[4][1] = {
  {
    new Tensor(3.5)
  },
  {
    new Tensor(12.0)
  },
  {
    new Tensor(10.0)
  },
  {
    new Tensor(9.5)
  },
};

const bool looping = false;

Layer* layer1;
Layer* layer2; 

void setup() {
  Serial.begin(9600);
  Serial.println("Training");
  randomSeed(analogRead(A0));
  
  layer1 = new Layer(2, 2);
  layer2 = new Layer(1, 2);

  const float lr = 0.01;
  const int epochs = 100;
  const int N = 4;

  for (int epoch = 0; epoch < epochs; epoch++) {
    float totalLoss = 0.0;

    for (int i = 0; i < N; i++) {
      Tensor** out1 = layer1->forward(xdata[i]);
      Tensor** out2 = layer2->forward(out1);

      Tensor* pred = out2[0];

      Tensor* loss = pred->mseLoss(ydata[i][0]);
      totalLoss += loss->getValue();

      loss->backward();

      layer1->updateWeights(lr);
      layer2->updateWeights(lr);

      layer1->zeroGradWeights();
      layer2->zeroGradWeights();

      loss->freeGraph();
      delete[] out1;
      delete[] out2;
    }

    Serial.print("Epoch ");
    Serial.print(epoch + 1);
    Serial.print("  loss: ");
    Serial.println(totalLoss / N, 6);
  }

  Serial.println("\n--- Hasil Prediksi ---");
  for (int i = 0; i < N; i++) {
    Tensor** out1 = layer1->forward(xdatatest[i]);
    Tensor** out2 = layer2->forward(out1);
    Serial.print("prediksi: ");
    Serial.print(out2[0]->getValue(), 4);
    Serial.print("  target: ");
    Serial.println(ydatatest[i][0]->getValue(), 4);
    out2[0]->freeGraph();

    delete[] out1;
    delete[] out2;
  }

  Serial.println("Selesai");

  if(looping){
    Serial.println("\n--- Loop Prediksi ---");
  }
}

float a = 1.0;
float b = 2.0;
float c = 3.0;

void loop() {
  if(!looping){
    return;
  }
  Tensor* input[2] = {new Tensor(a), new Tensor(b)};

  Tensor** out1 = layer1->forward(input);
  Tensor** out2 = layer2->forward(out1);

  Serial.print("prediksi: ");
  Serial.print(out2[0]->getValue(), 4);
  Serial.print("  target: ");
  Serial.println(c, 4);
  out2[0]->freeGraph();

  a += 1.0;
  b += 1.0;
  c += 1.0;

  delete input[0];
  delete input[1];

  delete[] out1;
  delete[] out2;

  delay(1000);
}
