import q2

hashmap={'1':'lrls','2':'onFold','3':'kFold'};

def main():
    while 1:
        num=input('Please select the test func: 1 for LRLS, 2 for on_fold, 3 for k_fold\n');
        if int(num) not in range(1,4):
            print ('Number input out of range, select again');
            continue;
        else:
            q2.test(hashmap[num]);

if __name__ == '__main__':
    main();