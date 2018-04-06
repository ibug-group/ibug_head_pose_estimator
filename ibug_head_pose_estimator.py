import os
import numpy as np


class HeadPoseEstimator:
    def __init__(self):
        # Load the model
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'head_pose.model')
        with open(model_path, 'r') as model_file:
            model_content = [float(x) for x in model_file.read().split(' ') if len(x) > 0]
            self._coefficients = self._parse_matrix(model_content)
            self._mean_shape = self._parse_matrix(model_content)
            del(model_content[0])
            mean = self._parse_matrix(model_content)
            self._parse_matrix(model_content)
            weights = self._parse_matrix(model_content)
            self._pitch_regressor = (mean, weights)
            mean = self._parse_matrix(model_content)
            self._parse_matrix(model_content)
            weights = self._parse_matrix(model_content)
            self._yaw_regressor = (mean, weights)

    def estimate_head_pose(self, landmarks):
        assert landmarks.shape[0] == 49 and landmarks.shape[1] == 2
        print(self.compute_rigid_alignment_parameters(landmarks, self._mean_shape))

    @staticmethod
    def _parse_matrix(model_content):
        shape = (int(model_content[0]), int(model_content[1]))
        matrix = np.array(model_content[2: 2 + shape[0] * shape[1]]).reshape(shape, order='F')
        del(model_content[0: 2 + shape[0] * shape[1]])
        return matrix

    @staticmethod
    def compute_rigid_alignment_parameters(source, destination):
        assert source.shape == destination.shape and source.shape[1] == 2
        a = np.zeros((4, 4), np.float)
        b = np.zeros((4, 1), np.float)
        for idx in range(source.shape[0]):
            a[0, 0] += np.dot(source[idx], source[idx])
            a[0, 2] += source[idx, 0]
            a[0, 3] += source[idx, 1]
            b[0] += np.dot(source[idx], destination[idx])
            b[1] += np.dot(source[idx], [destination[idx, 1], -destination[idx, 0]])
            b[2] += destination[idx, 0]
            b[3] += destination[idx, 1]
        a[1, 1] = a[0, 0]
        a[3, 0] = a[0, 3]
        a[1, 2] = a[2, 1] = -a[0, 3]
        a[1, 3] = a[3, 1] = a[2, 0] = a[0, 2]
        a[2, 2] = a[3, 3] = source.shape[0]
        return tuple(*np.dot(np.linalg.pinv(a), b).T)


def test_head_pose_estimator():
    estimator = HeadPoseEstimator()
    # print(estimator._mean_shape)
    # print(estimator._coefficients)
    # print(estimator._pitch_regressor)
    # print(estimator._yaw_regressor)
    print('Head pose estimator initialised.')
    test_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
    landmark_files = sorted(glob.glob(os.path.join(test_data_folder, '*.txt')))
    head_pose_estimations = np.zeros((len(landmark_files), 4), np.float)
    print('Now estimating head pose from the %d landmark files in "%s"...' %
          (len(landmark_files), test_data_folder))
    for idx in range(len(landmark_files)):
        landmark_file_path = landmark_files[idx]
        head_pose_estimations[idx, 0] = os.path.splitext(os.path.basename(landmark_file_path))[0]
        with open(landmark_file_path, 'r') as landmark_file:
            numbers = [float(x) for x in str.replace(landmark_file.read(), '\n', ' ').split(' ') if len(x) > 0]
            landmarks = np.array(numbers[3: 101]).reshape((-1, 2))
            head_pose_estimations[idx, 1:] = estimator.estimate_head_pose(landmarks)
        break
    output_file_path = os.path.join(test_data_folder, 'head_pose.csv')
    np.savetxt(output_file_path, head_pose_estimations, delimiter=',', comments='',
               fmt=['%d'] + ['%.3f'] * 3, header='frame_number,pitch,yaw,roll')
    print('Done, results are saved in "%s".' % output_file_path)


if __name__ == '__main__':
    import glob
    test_head_pose_estimator()
