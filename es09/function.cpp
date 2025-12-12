#include "function.h" 
#include <limits>

#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */


void progress(int step, int nstep, int barlenght)
{
	int l=barlenght*(1+float(step))/nstep;
	string str="[";
	for (int b=0; b<l; b++) str+="|";
	cout<<GREEN <<"\r\t"<<str<<setw(barlenght-l+1)<<"]["
      <<fixed<<setprecision(2) <<100*l/float(barlenght)<<"%]"<<flush;
    cout<<RESET;
} 



void print(crom s, string prefactor)
{
    for(int k:s)
        cout <<prefactor<< k << " ";
    cout<<endl;
    return;
}

void print(vector<crom> &pop, string prefactor)
{
    for(crom s:pop)
        print(s, prefactor);
    return;
}



void init_gen(
    Random &rnd, string primesfile, 
    int primerow, string seedfile)
{

    // INIT RANDOM GENERATOR
    int p1, p2; // Read from ../INPUT/Primes a pair of numbers to be used to initialize the RNG
    ifstream Primes(primesfile);
    for(int r=0; r<primerow-1; r++)
        Primes.ignore (
            std::numeric_limits<std::streamsize>::max(),
            '\n' 
        ); //skip primerow lines 
    Primes >> p1 >> p2 ;
    Primes.close();
    int seed[4]; // Read the seed of the RNG
    ifstream Seed(seedfile);
    Seed >> seed[0] >> seed[1] >> seed[2] >> seed[3];
    
    rnd.SetRandom(seed,p1,p2);
    return;
}

