
import tensorflow as tf
import os
from object_detection import model_lib
from object_detection import model_hparams


def train_pet():
    # Create RunConfig
    home_dir = os.getenv('HOME')
    pwd_path = os.path.dirname(os.path.realpath(__file__))

    model_dir = os.path.join(home_dir, 'workspace/cv_models/obj_det/pet')
    gpu_options = tf.GPUOptions(allow_growth=True)
    train_distribute = tf.distribute.MirroredStrategy(
        devices=["/device:GPU:0", "/device:GPU:1"]
    )
    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        session_config=tf.ConfigProto(gpu_options=gpu_options),
        #train_distribute=train_distribute,
    )

    # Create Estimator and so on
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path=os.path.join(pwd_path, 'faster_rcnn_resnet101_pets.config'),
    )
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == '__main__':
    train_pet()