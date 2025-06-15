#include "func.h"


double error( const vector<double> &AV, const vector<double> &AV2, int n )
{
	/*
	Function for statistical uncerainty estimation
	*/
	if(n==0) //no unc for first data
		return 0.;
	else
		return sqrt((AV2[n] - pow(AV[n], 2) )/n); 
}

int* complement_seeds(
	string filename, //="/home/tommaso/LSN/Parallel_number_generator_20250306/Primes",
	int row //= 0
)
{
	/*
	Read complement seed from specified file
	*/
	
	//cout<<filename<<endl;
	

	int p1, p2;
	ifstream Primes(filename);
	// load first 2 numbers in primes file into complement seeds
	// to generate a statistical indipendent new sequence read a new line of primes
	if (Primes.is_open()){
		int counter=0;
		do
		{
			Primes >> p1 >> p2 ;
			counter+=1;
		} while (counter<row);
		
	} else cerr << "PROBLEM: Unable to open "<<filename << endl;
	Primes.close();
	
	static int v[2]={p1,p2};
	
	//cout<<"Primes seeds: "<<v[0]<< " "<<v[1]<<endl;
	return v;
}

void set_seed(
	Random &rnd,
	string seedfile,//="/home/tommaso/LSN/Parallel_number_generator_20250306/seed.in",
	string primesname//="/home/tommaso/LSN/Parallel_number_generator_20250306/Primes"
)
{


   int seed[4];
   int* p = complement_seeds(primesname);
   //cout<<p<<endl;
   

   ifstream input(seedfile);
   string property;
   if (input.is_open()){
      while ( !input.eof() ){
         input >> property;
         if( property == "RANDOMSEED" ){
            input >> seed[0] >> seed[1] >> seed[2] >> seed[3];
            rnd.SetRandom(seed,p[0],p[1]);
         }
      }
      input.close();
   } else cerr << "PROBLEM: Unable to open "<<seedfile << endl;
   
   cout<<"Random generator parameters:|"<<endl
       <<"____________________________|"<<endl
       <<"\t| seed        : "<<seed[0]<<seed[1]<<seed[2]<<seed[3]<<endl
       <<"\t| Primes seeds: "<<p[0]<<", "<<p[1]<<endl<<endl;
   
   //delete p;

}


void set_numeric_seed(
	Random &rnd,
	int seed[4],
	string primesname
)
{


   int* p = complement_seeds(primesname);
   //cout<<p<<endl;
   

    rnd.SetRandom(seed,p[0],p[1]);
   
   cout<<"Random generator parameters:|"<<endl
       <<"____________________________|"<<endl
       <<"\t| seed        : "<<seed[0]<<seed[1]<<seed[2]<<seed[3]<<endl
       <<"\t| Primes seeds: "<<p[0]<<", "<<p[1]<<endl<<endl;
   
   //delete p;

}


void save_data(
	vector<int> &throws, 
	vector<double> &data, 
	vector<double> &err,  
	string filename//=""
)
{
	if(data.size()!=err.size())
	{	
		cerr<<"Error: Unmatching dimension between data and error"<<endl;
		cerr<<"data dimension "<<data.size()<<", error dimension "<<err.size()<<endl;
		exit(1);
	
	}
	
	if(data.size()!=throws.size())
	{
		cerr<<"Error: Unmatching dimension between data and number of throw"<<endl;
		cerr<<"data dimension "<<data.size()<<", throw dimension "<<throws.size()<<endl;
		exit(1);

	}
	
	int size=data.size();
	
	//default value for filename
	if (filename=="")
	{
		filename="data.csv";
	}	
	
	ofstream out(filename);
	
	if (!out.is_open())
	{
		cerr<<"Error: can not open "<<filename<<endl;
		exit(1);
	}
	
	out<<"throws,avg,err"<<endl;
	
	for (int i=0; i<size; ++i)
	{	
		out<<throws[i]<<", "<<data[i]<<", "<<err[i]<<endl;
	}
	
	out.close();
	cout<<"Data saved in path "<<filename<<endl;
}



void save_data(
	vector<float> &data, 
	string filename
)
{
	int size=data.size();
	
	//default value for filename
	if (filename=="")
	{
		filename="data.csv";
	}	
	
	ofstream out(filename);
	
	if (!out.is_open())
	{
		cerr<<"Error: can not open "<<filename<<endl;
		exit(1);
	}
	
	out<<"data"<<endl;
	
	for (int i=0; i<size; ++i)
	{	
		out<<data[i]<<endl;
	}
	
	out.close();
	cout<<"Data saved in path "<<filename<<endl;
}


void save_data(
	vector<double> &data, 
	string filename
)
{
	int size=data.size();
	
	//default value for filename
	if (filename=="")
	{
		filename="data.csv";
	}	
	
	ofstream out(filename);
	
	if (!out.is_open())
	{
		cerr<<"Error: can not open "<<filename<<endl;
		exit(1);
	}
	
	out<<"data"<<endl;
	
	for (int i=0; i<size; ++i)
	{	
		out<<data[i]<<endl;
	}
	
	out.close();
	cout<<"Data saved in path "<<filename<<endl;
}



void progress(int block, int nblock, int barlenght=30)
{
	int l=barlenght*float(block)/nblock;
	string str="[";
	for (int b=0; b<l; b++) str+="=";
	cout<<"\r"<<str<<">"<<setw(barlenght-l)<<"]["<<100*l/float(barlenght)<<"%]"<<flush;
} 

