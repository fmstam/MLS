import sys
sys.path.append("..")

from MLS.examples.all_one import scenario as s



def main():
    s.env.step(5)
    print(s.env.play_ground)


if __name__ == "__main__":
    # execute only if run as a script
   main()
