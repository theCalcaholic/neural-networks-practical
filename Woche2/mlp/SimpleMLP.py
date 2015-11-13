# -*- coding: utf-8 -*-

import numpy

class MLP:
  def __init__(self):
    self.W_1 = None
    # self -> verweis auf das aktuelle exemplar
    self.W_2 = None
    self.trained = False

  def train(self, data_in, data_out):
    # Train -> Kantengewichte Trainieren
    # Eingabelayer
    size_in = numpy.shape(data_in)[1]
    # Hiddenlayer, 3 Knoten
    size_hid = 3

    # size of layer 1 (output layer)
    size_out = numpy.shape(data_out)[1]
    # Ausgabelayer
    # allocate weight matrix from layer 0 to layer 1
    # 2 Matritzen, zufällige Werte zw. -1 und 1
    self.W_1 = numpy.random.uniform(-1.0, 1.0, (size_hid, size_in))
    self.W_2 = numpy.random.uniform(-1.0, 1.0, (size_out, size_hid))
    # reihenfolge_
    # bias_1 = numpy.zeros(size_hid)
    # ab hier schleife
    for t in range (10000):
      cost = 0.0
      for i in range(len(data_in)):
        # range? len?
        results = self._feedforward(data_in[i])

        # Backpropagation
        T_1 = data_out[i]
        # T_1 das Zielergebnis
        delta_out = T_1 - results["out"]
        # differenz zwischen t_1 und dem tatsächlichen ergbn. -> (|delta_out|=fehler)
        delta_hid = (results["hidden_out"] * (1 - results["hidden_out"])) * numpy.dot(delta_out, self.W_2)
        # Abl. von Aktivierungsfunktion (1- H_out), delta_out ist der Fehler
        delta_W1 = 0.1 * numpy.outer(delta_hid, data_in[i])
        self.W_1 = self.W_1 + delta_W1
        delta_W2 = 0.1 * numpy.outer(delta_out, results["hidden_out"])
        self.W_2 = self.W_2 + delta_W2

        #print "S=", S, "T_1=", T_1, "delta=", delta ?

        cost += delta_out**2

      print "cost=", cost
    self.trained = True

  def _feedforward(self, data):
    # nimmt einen Eingabewert und gibt die Outputwerte von allen Layern zurück

    H_net = numpy.dot(self.W_1, data)# + bias_1
    # ergebnis der Matrizenmultiplikation W_1 und data_in
    H_out = 1/(1+(numpy.exp(-H_net)))
    # aktivierungsfunktion wird auf h_net angewendet
    S = numpy.dot(self.W_2, H_out)
    S = S
    # simple aktiv. funktion
    # Gewichte der Kanten = w,
    return {"out": S, "hidden_out": H_out}

  def predict(self, data):
    # nimmt einen Wert und berechnet dafür den Output
    if not self.trained:
      raise Exception("MLP has not been trained yet")
    return self._feedforward(data)["out"]
