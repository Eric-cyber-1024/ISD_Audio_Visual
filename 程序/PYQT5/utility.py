import sounddevice as sd


def audio_dev_to_str(device):
    '''
    get info of a sounddevice device
    '''
    hostapi_names = [hostapi['name'] for hostapi in sd.query_hostapis()]
    text = u'{name}, {ha} ({ins} in, {outs} out)'.format(
        name=device['name'],
        ha=hostapi_names[device['hostapi']],
        ins=device['max_input_channels'],
        outs=device['max_output_channels'])
    return text
