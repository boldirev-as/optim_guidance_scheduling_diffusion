from torchvision.models import inception_v3, Inception_V3_Weights

inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True, transform_input=False)
