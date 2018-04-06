import os
import numpy as np


class HeadPoseEstimator:
    def __init__(self):
        # Load the model
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'head_pose.model')
        with open(model_path, 'r') as model_file:
            model_content = [float(x) for x in model_file.read().split(' ') if len(x) > 0]
            head = 0
            self._coefficients, head = self._parse_matrix(model_content, head)
            self._mean_shape, head = self._parse_matrix(model_content, head)
            head += 1
            mean, head = self._parse_matrix(model_content, head)
            _, head = self._parse_matrix(model_content, head)
            weights, head = self._parse_matrix(model_content, head)
            self._yaw_regressor = (mean, weights.reshape((1, -1), order='F')[0])
            mean, head = self._parse_matrix(model_content, head)
            _, head = self._parse_matrix(model_content, head)
            weights, head = self._parse_matrix(model_content, head)
            self._pitch_regressor = (mean, weights.reshape((1, -1), order='F')[0])

    def estimate_head_pose(self, landmarks):
        assert landmarks.shape[0] == 49 and landmarks.shape[1] == 2
        aligned_upper_face = self.apply_rigid_alignment_parameters(
            landmarks[0: 31, :], *self.compute_rigid_alignment_parameters(landmarks[0: 31, :],
                                                                          self._yaw_regressor[0]))
        yaw = (np.dot(aligned_upper_face.reshape((1, -1), order='F'), self._yaw_regressor[1][0: -1]) +
               self._yaw_regressor[1][-1])[0] * 1.25
        pitch = (np.dot(aligned_upper_face.reshape((1, -1), order='F'), self._pitch_regressor[1][0: -1]) +
                 self._pitch_regressor[1][-1])[0] * 1.25
        scos, ssin, _, _ = self.compute_rigid_alignment_parameters(landmarks, self._mean_shape)
        roll = np.degrees(np.arctan2(ssin, scos))
        return tuple(np.dot(np.array([pitch, yaw, roll]), self._coefficients).tolist())

    @staticmethod
    def _parse_matrix(model_content, head):
        shape = (int(model_content[head]), int(model_content[head + 1]))
        head += 2
        matrix = np.array(model_content[head: head + shape[0] * shape[1]]).reshape(shape, order='F')
        return matrix, head + shape[0] * shape[1]

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
        return tuple(np.dot(np.linalg.pinv(a), b).T[0].tolist())

    @staticmethod
    def apply_rigid_alignment_parameters(source, scos, ssin, transx, transy):
        return np.dot(source, [[scos, ssin], [-ssin, scos]]) + [transx, transy]


def test_head_pose_estimator():
    estimator = HeadPoseEstimator()
    print('Head pose estimator initialised.')
    test_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data')
    landmark_files = sorted(glob.glob(os.path.join(test_data_folder, '*.txt')))
    head_pose_estimations = np.zeros((len(landmark_files), 4), np.float)
    print('Now estimating head pose from the %d landmark files in "%s"...' %
          (len(landmark_files), test_data_folder))
    start_time = time.time()
    for idx in range(len(landmark_files)):
        landmark_file_path = landmark_files[idx]
        head_pose_estimations[idx, 0] = os.path.splitext(os.path.basename(landmark_file_path))[0]
        with open(landmark_file_path, 'r') as landmark_file:
            numbers = [float(x) for x in str.replace(landmark_file.read(), '\n', ' ').split(' ') if len(x) > 0]
            landmarks = np.array(numbers[3: 101]).reshape((-1, 2))
            head_pose_estimations[idx, 1:] = list(estimator.estimate_head_pose(landmarks))
    output_file_path = os.path.join(test_data_folder, 'head_pose.csv')
    np.savetxt(output_file_path, head_pose_estimations, delimiter=',', comments='',
               fmt=['%d'] + ['%.3f'] * 3, header='frame_number,pitch,yaw,roll')
    print('Done in %.3f second, results are saved in "%s".' % (time.time() - start_time, output_file_path))


if __name__ == '__main__':
    import glob
    import time
    test_head_pose_estimator()
