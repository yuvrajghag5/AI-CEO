from automate import full
import main as dashboard
def main():
    
    full.main()
    print("\n" + "=" * 50)
    print("dashboard STARTING NOW !! \n")
    print("=" * 50)
    dashboard.main()



if __name__ == "__main__":
    main()