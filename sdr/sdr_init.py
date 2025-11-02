import adi


def initialize_pluto_sdr(config):
    sdr = adi.Pluto(config["uri"])
    sdr.sample_rate = int(config["sample_rate"])
    sdr.rx_lo = int(config["center_frequency"])
    sdr.rx_rf_bandwidth = int(config["bw"])
    sdr.rx_hardwaregain_chan0 = int(config["rx_hardwaregain_chan0"])
    sdr.gain_control_mode_chan0 = config["gain_control_mode_chan0"]
    sdr.rx_buffer_size = config["rx_buffer_size"]
    sdr.tx_destroy_buffer()

    return sdr
