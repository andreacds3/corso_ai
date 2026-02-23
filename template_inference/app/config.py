from typing import Optional

import yaml
from pydantic import BaseModel

from schema.base import Device


class BatchConfiguration(BaseModel):
    """Represent the configuration of the Batcher.
    :param batch_size the number of texts to be summarized in batch
    :param batch_timeout the number of seconds to wait for considering a batch 'closed' and proceeding to summarization"""
    batch_size: int | None = 4
    batch_timeout: float | None = 1.0


class Configuration(BaseModel):
    """
    Represent the configuration of the whole application.
    :param batch_configuration the configuration parameters of the batcher
    :param device what is the device where the model should be loaded ('cpu' or 'cuda')
    :param default_max_length: Default maximum number of tokens for summary generation
    :param default_min_length: Default minimum number of tokens for summary generation
    """
    device: Device
    default_min_length: int
    default_max_length: int
    batch_configuration: Optional[BatchConfiguration]

    @classmethod
    def from_configuration_file(cls, config_file_path):
        """
        Load the yaml configuration file and initialize the configuration object
        :param config_file_path: the path of the configuration yaml file
        :return: the configuration object
        """
        with open(config_file_path, 'r') as file:
            configuration = yaml.safe_load(file)

        batch = configuration['batch']
        if batch:
            batch_size = configuration['batch_size'] if 'batch_size' in configuration else None
            batch_timeout = configuration['batch_timeout'] if 'batch_timeout' in configuration else None
            batch_conf = BatchConfiguration(batch_size=batch_size, batch_timeout=batch_timeout)
        else:
            batch_conf = None

        device = Device(configuration['device']) if 'device' in configuration else Device('cpu')
        default_min_length = configuration['default_min_length'] if 'default_min_length' in configuration else 0
        default_max_length = configuration['default_max_length'] if 'default_max_length' in configuration else 512
        return cls(
            device=device,
            default_min_length=default_min_length,
            default_max_length=default_max_length,
            batch_configuration=batch_conf
        )


if __name__ == '__main__':
    print(Configuration.from_configuration_file('../configuration.yml'))
