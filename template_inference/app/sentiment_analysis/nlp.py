from typing import List, Tuple
import asyncio
import torch
from sentiment_analysis.model import RNNClassifier
from schema.base import Device
from logging_utils import APPLICATION_LOGGER
import json

class NLPSentimentPipeline:
    """Class representing a full summarization pipeline"""

    def __init__(self, device: Device):
        """
        Initialize the NLPSentimentPipeline
        :param device: The device to load the model on (cpu or cuda)
        """
        self._device = device
        self._model_params = '/model/miglior_modello_imdb.pt'
        self._vocab = None
        self.model = None
        self.tokenizer = None
        self.CLASSES_DICT = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

    def initialize(self) -> None:
        """Initialize the pipeline by loading the appropriate model and tokenizer"""
        self.model = RNNClassifier(25002, 100, 64, 1, 0.5)
        self.model.load_state_dict(torch.load(self._model_params))
        with open('/model/vocab.json') as json_data:
            self._vocab = json.load(json_data)

    async def extract_sentiment(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Extract sentiment from the given texts
        :param texts: The list of texts to be summarized (a batch)
        :return: The sentiment and the
        """

        APPLICATION_LOGGER.debug(f'Received classification request for {len(texts)} documents')

        scores = self.predict_sentiment(texts)
        predictions = ['POSITIVE' if score > 0.5 else 'NEGATIVE' for score in scores]
        await asyncio.sleep(0)


        return [(prediction, score) for score, prediction in zip(scores, predictions)]

    def predict_sentiment(self, sentences):
        self.model.eval()  # Importante: spegne il Dropout per avere risultati stabili

        # 1. Tokenizzazione (uguale a quella fatta nel training)
        tokens = [sentence.lower().split() for sentence in sentences]

        # 2. Convertiamo parole in numeri
        # Se la parola non è nel vocabolario, usiamo l'indice 1 (<UNK>)
        indexed = [[self._vocab.get(t, 1) for t in token_single] for token_single in tokens]

        # 3. Trasformiamo in Tensore PyTorch
        tensor = torch.LongTensor(indexed).to(self._device)

        # 5. Predizione
        with torch.no_grad():
            prediction = self._model(tensor)

        # Restituiamo i valori float (da 0 a 1)
        return prediction.item()