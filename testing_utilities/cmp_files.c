#include <stdio.h>
#include <string.h>

struct Classes {
    char        name[100];
    __uint32_t  qty;
};

int main () {
    FILE *p_file_list;
    FILE *p_file_results;
    FILE *p_file_comparison;
    FILE *p_file_classes_names;
    char str_file_name[100];
    char buffer[100];
    char buffer_2[100];
    char *buffer_aux;
    __uint32_t idx_var = 0u;


    // 0) Open the .txt file with the a lists of images names
    p_file_list = fopen("file_list.txt", "r");
    if (p_file_list == NULL) 
    {
        printf("Could not open file file_list.txt\n");
        return (0u); 
    }

    p_file_comparison = fopen("results_comparison.csv", "w+");
    if (p_file_comparison == NULL) 
    {
        printf("Could not open file results_comparison.csv\n");
        return (0u); 
    }

    // 1) Iterate through all elements in the list
    size_t ui32_num_list_names = 2u;
    for (size_t idx_loop = 0; idx_loop < ui32_num_list_names; idx_loop++)
    {
        // =====================================================================
        // 0) Extract and modify the file name from a list
        // =====================================================================
        // Extract the file name and store in buffer
        fscanf(p_file_list,"%s",buffer);
        // printf("iteration:%i, name=%s\n", idx_loop,buffer);

        // Delete first "0" character from the buffer
        memmove(buffer, buffer+1, strlen(buffer));

        // Modify the name (include the extension and the iteration in terms of execution order)
        snprintf(str_file_name, 100,"%ld_%s.csv",idx_loop,buffer);
        printf("%s\n",str_file_name);



        // ======================================================================================
        // 1) Now i am going to count the number of detected clases and store then in 2 arrays:
        //      a) First one has the name of the detected clases
        //      b) Second on has the number of detected clases, respectively
        // CLUE: https://www.tutorialspoint.com/c_standard_library/c_function_strtok.htm
        // ======================================================================================
        // Open the .csv file with the results
        p_file_results = fopen(str_file_name, "r");
        if (p_file_results == NULL) 
        {
            printf("Could not open file %s\n",str_file_name);
            return (0u); 
        }

        // Struct to store the number of classes detected (we are detecting with this config 80 different classes)
        struct Classes classes_array[80u];
        struct Classes classes_golden_array[80u];

        //p_file_classes_names = fopen("cfg/coco.names", "r");
        p_file_classes_names = fopen("../cfg/coco.names", "r");
        if (p_file_classes_names == NULL) 
        {
            printf("Could not open file coco.names\n");
            return (0u); 
        }

        // Store in struct classes the classes detected (name) and the quantity detected (qty)
        __uint32_t idx_data_names = 0u;
        while (!feof(p_file_classes_names))
       {
            // Extract the file name and store in buffer
            fscanf(p_file_classes_names,"%[^\n] ",buffer_2);
            strcpy(classes_array[idx_data_names].name,buffer_2);
            classes_array[idx_data_names].qty=0u;
            printf("%i) class:%s qty:%i\n", idx_data_names, classes_array[idx_data_names].name,classes_array[idx_data_names].qty);
            idx_data_names++;
        }
        fclose(p_file_classes_names);
        
        // Move the pointer to the beggining of the file
        fseek(p_file_results,0,SEEK_SET);
        
        // Skip the first line
        fscanf(p_file_results,"%[^\n] ",buffer);
        __uint32_t detected_classes;
        detected_classes = 0u;
        while(!feof(p_file_results))
        {
            detected_classes++;
            fscanf(p_file_results,"%[^\n] ",buffer);
            // printf("buffer:%s\n",buffer);
            buffer_aux = strtok(buffer,"_");
            __uint32_t idx_clasess;
            idx_clasess  = 0u;
            // printf("buffer aux:%s\n",buffer_aux);

            for (size_t idx_clasess = 0; idx_clasess < 20u; idx_clasess++)
            {
                if (strcmp("0",classes_array[idx_clasess].name)==0)
                {
                    strcpy(classes_array[idx_clasess].name,buffer_aux);
                    classes_array[idx_clasess].qty++;
                    break;
                }
                else if (strcmp(buffer_aux,classes_array[idx_clasess].name)==0)
                {
                    classes_array[idx_clasess].qty++;
                    break;
                }
            }
        }
        
        for (size_t idx_clasess = 0; idx_clasess < 5; idx_clasess++)
        {
            }
        


        // Store the results in a .csv file
        fprintf(p_file_comparison,"%s,%i",str_file_name,detected_classes);
        for (size_t idx_clasess = 0; idx_clasess < 80; idx_clasess++)
        {
            fprintf(p_file_comparison,",%d",classes_array[idx_clasess].qty);
            printf("class:%s\tquantity:%d\n",classes_array[idx_clasess].name,classes_array[idx_clasess].qty);
        }
        fprintf(p_file_comparison,"\n");
        printf("%s\n",buffer);

    }
    fclose(p_file_results);
    fclose(p_file_list);
    fclose(p_file_comparison);
    
   
    return(0);
}



        // Count the number of classess detected (is the same minus one the number of lines in the .csv file)
        // __int32_t lines;
        // lines = 0u;
        // char ch ;
        // // fscanf(p_file_results,"%s",buffer);
        // while (!feof(p_file_results))
        // {
        //     ch = fgetc(p_file_results);
        //     if(ch == '\n')
        //     {
        //         lines++;
        //     }
        // }
        // // Remember to substract 1u to final lines values
        // lines--;