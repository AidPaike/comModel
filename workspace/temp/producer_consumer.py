from multiprocessing.pool import ThreadPool
import subprocess

combine_cmd = ["llvm-profdata-10 ","merge","-sparse","/root/comModel/data/coverage/Finetune_distilgpt2/checkpoint-20000/55582608c16911ed89670242ac180002.js.profdata",
               "/root/comModel/data/coverage/Finetune_distilgpt2/checkpoint-20000/55857536c16911edaf750242ac180002.js.profdata",
                "-o","/root/comModel/data/coverage/Finetune_distilgpt2_checkpoint-20000.profdata","&&","ls","-l"]


combine_cmd = "bash"
cmds = ['echo start', 'echo mid', 'echo end']
# combine_cmd = ["du","-lh",";","ls","-l"]
pro = subprocess.Popen("/bin/bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False,
                                            stderr=subprocess.PIPE, universal_newlines=True)
commands = "du -lh \n cd report \n ls"
            
# for cmd in commands:
#     pro.stdin.write(cmd+"\n")
# print(pro.stdin.read())
# pro.stdin.close()

stdout, stderr = pro.communicate(commands)
print(stdout)
print(stderr)