import argparse

def parse(args=None):
    parser = argparse.ArgumentParser(description='hyd')

    # basic config
    parser.add_argument('--device', type=int, required=False, default= 2,
                        help='GPU ID')
    parser.add_argument('--task_name', type=str, required=False, default='uni-component',
                        help='task name, options:[uni-component, bi-component]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    
    # data
    parser.add_argument('--dataset', type=str, required=False, default= 'cycDEH', choices=['cycDEH','isoDEH','TPD',
                                                                                        'bi-isoDEH','bi-cycDEH','bi-TPD'])
    parser.add_argument('--num_data', type=int, default=0, help='')
    parser.add_argument('--independent_v', type=int, default=12, help='setting the x axis')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--remove_y', type=bool, default=False, help='whether need remove y')
    parser.add_argument('--data_mode', type=str, default= 'one-to-one', choices=['multi-to-one','one-to-one'])
    parser.add_argument('--test_data', type=str, default= 'real', help='whether is real test data ,or generated data')

    # some path
    parser.add_argument('--res_path', type=str, default='res/', help='')
    parser.add_argument('--res_fig', type=str, default='fig/', help='')
    parser.add_argument('--res_pdf', type=str, default='pdf/', help='')
    
    
    # parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=False, default='RandomForestRegressor',
                        help='model name')  #RandomForestRegressor,Attention,TransformerRegressor
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='whether data shuffle befor data splitation')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size') # default=1
    parser.add_argument('--win', type=int, default=2, help='windows size')  # if data_mode == 'one-to-one', win==1, else win == 4
    parser.add_argument('--win_step', type=int, default=1, help='windows step')
      
    # model
    parser.add_argument('--input_size', type=int, default=5, help='feature of input')
    parser.add_argument('--hidden_size', type=int, default=8, help='feature of hidden state')
    parser.add_argument('--output_size', type=int, default=1, help='feature of output')
    parser.add_argument('--num_layers', type=int, default=1, help='') # default=2
    parser.add_argument('--num_heads', type=int, default=2, help='')   # if data_mode == 'one-to-one', num_heads==1, else num_heads == 2
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate') # default=0.0001
    args = parser.parse_args()

    return args