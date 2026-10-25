from models.pose_classifier import PoseClassifier

def test_classifier(): assert PoseClassifier().predict([]) == 'nominal'
