from TicTacToeEnv import TicTacToeEnv

def main():
    env = TicTacToeEnv()


    while(True):
        env.reset()
        done = False

        while not done:
            print("\033[H\033[2J")
            env.render()
            s, r, done, info = env.step(int(input("? ")))

        print("\033[H\033[2J")
        env.render()
        print(info)

        if(input("again [y/n]? ").upper() != 'Y'):
            break

if __name__ == '__main__':
    main()

