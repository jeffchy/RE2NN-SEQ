from src_seq.utils import mkdir, create_datetime_str
from itertools import product
import os


def gen_commands_last(args):
    commands = []

    values = args.values()
    keys = list(args.keys())

    iter = product(*values)
    for i in iter:
        assert len(i) == len(keys)
        cmd = ''

        for j in range(len(keys)):
            param = keys[j]
            param_val = i[j]

            cmd += ' --{} {}'.format(param, param_val)

        commands.append(cmd)
    print('TOTAL NUMBER OF COMMANDS: {}'.format(len(commands)))
    return commands


def gen_scripts_file_and_macro_commands(commands, gpus_info, env_info):
    datetime_str = create_datetime_str()

    header = 'cd ../../src_seq'

    macro_commands = {
        k: [] for k, v in gpus_info.items()
    }
    files_number = sum([len(i) for i in gpus_info.values()])
    mkdir(env_info['script_dir'])
    avg_commands = int(len(commands) / files_number) + 1
    fn = 0
    for k, gpus in gpus_info.items():
        for gpu in gpus:
            file_name = "{}.{}.{}.{}.sh".format(env_info['name'], datetime_str, gpu, fn)
            fpath = os.path.join(env_info['script_dir'], file_name)
            left = fn * avg_commands
            right = min((fn + 1) * avg_commands, len(commands))

            with open(fpath, 'w', encoding='utf-8') as f:
                f.writelines(header)
                f.writelines('\n')

                for cmd in commands[left: right]:
                    cmd_complete = "{}={} {} {} {}".format(
                        env_info['cuda_str'], gpu, env_info['python_env'], env_info['main_name'], cmd
                    )
                    f.writelines(cmd_complete)
                    f.writelines('\n')
            fn += 1
            macro_commands[k].append(file_name)

    return macro_commands


def gen_final_scripts(macro_commands, env_info):
    l = ''

    for k, v in macro_commands.items():
        l += 'sleep 3\n'
        l += 'echo "START RUNNING IN NODE{}"\n'.format(k)
        l += 'ssh node{}\n'.format(k)
        l += 'hostname\n'
        l += 'cd {}\n'.format(env_info['script_dir'])
        for scripts in v:
            l += 'nohup sh {} > {} & \n'.format(scripts, 'out.' + scripts[:-3])

    datetime_str = create_datetime_str()
    with open(os.path.join(env_info['script_dir'], 'run.{}.sh'.format(datetime_str)), 'w') as f:
        f.write(l)

    print(l)