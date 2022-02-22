#include <stdio.h>
#include <string.h>
#include <stdlib.h> // atoi()
#include <math.h>   // abs()

struct Classes {
    char        name[100];
    __int32_t  accuracy;
    __int32_t   left_x;
    __int32_t   top_y;
    __int32_t  width;
    __int32_t  height;
    __int32_t  compared;
};

struct Pred_comp {
    __uint32_t golden_classes;
    __uint32_t expected_detection;
    __uint32_t fi_classes;
    __uint32_t undetected_classes;
    __uint32_t misclassification_accuracy;
    __uint32_t misclassification_widht;
    __uint32_t misclassification_height;
};

int main () {
    // Definition of the files involved in the comparison
    FILE *p_file_list;
    FILE *p_file_golden;
    FILE *p_file_fi_results;
    FILE *p_file_comparison;

    // Definition of variables
    char str_file_name[100];
    char str_golden_file_name[100];
    char buffer[100];
    char *buffer_aux;

    // Misclassification criteria variables
    __int32_t accuracy_range = 20u;
    __int32_t left_x_range   = 50u;
    __int32_t top_y_range    = 50u;
    __int32_t width_range    = 20u;
    __int32_t heigth_range   = 20u;


    // =====================================================
    // 0) Open the files and verify that can be opened
    // =====================================================
    p_file_list = fopen("file_list.txt", "r");
    if (p_file_list == NULL) 
    {
        printf("Could not open file file_list.txt\n");
        return (0u); 
    }


    // In this file we store the comparison results
    p_file_comparison = fopen("results_comparison.csv", "w+");
    if (p_file_comparison == NULL)
    {
        printf("Could not open file results_comparison.csv\n");
        return (0u);
    }

    // 1) Iterate through all elements in the list
    size_t ui32_num_list_names = 10000u;
    for (size_t idx_loop = 0; idx_loop < ui32_num_list_names; idx_loop++)
    {
        // =====================================================================
        // 2) Extract and modify the file name from a list (Adequacy)
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
        //      CLUE: https://www.tutorialspoint.com/c_standard_library/c_function_strtok.htm
        // ======================================================================================
        // Open the .csv file with the classification (after inject a bit flip)
        p_file_fi_results = fopen(str_file_name, "r+");
        if (p_file_fi_results == NULL) 
        {
            printf("Could not open file %s\n",str_file_name);
            return (0u); 
        }
        
        // Modify the path (include the folder root)
        snprintf(str_golden_file_name, 100,"golden_values/%s",str_file_name);

        // Open the .csv file with the results
        p_file_golden = fopen(str_golden_file_name, "r");
        if (p_file_golden == NULL) 
        {
            printf("Could not open file %s\n",str_golden_file_name);
            return (0u); 
        }

        // ======================================================================================
        // 2)   
        // ======================================================================================
        // Struct to store the number of classes detected:
        struct Pred_comp pred_comp;
        struct Classes golden_classes;
        struct Classes fi_classes[50u];

        // Store in struct "classes" the detected classes (from golden and from fi)
        // Move the pointer to the beggining of the file
        fseek(p_file_golden,0,SEEK_SET);
        fseek(p_file_fi_results,0,SEEK_SET);

        if (idx_loop==0)
        {
            fprintf(p_file_comparison,"image_name,golden_classes(exp),detected_classes,correct_detections,accuracy (mis),width (misc),height (miss),undetected_classes\n");
        }
        
        
        // Skip the first line
        fscanf(p_file_golden,"%[^\n] ",buffer);
        fscanf(p_file_fi_results,"%[^\n] ",buffer);

        // Initialize to 0 all values in pred_comp
        pred_comp.misclassification_widht = 0u;         // Falla la anchura
        pred_comp.misclassification_height = 0u;        // falla la altura
        pred_comp.misclassification_accuracy = 0u;      // falla la precisión
        pred_comp.golden_classes = 0u;                   // Número de detecciones en golden
        pred_comp.fi_classes = 0u;                      // Número de detecciones en fi_result
        pred_comp.expected_detection = 0u;              // Número de detecciones correctas
        pred_comp.undetected_classes = 0u;              // Número de clases de golden no detectadas en fi_results

        // Store in an array the results from the predictions with fi
        while(!feof(p_file_fi_results))
        {
            fscanf(p_file_fi_results,"%[^\n] ",buffer);
            buffer_aux = strtok(buffer,"_");
            strcpy(fi_classes[pred_comp.fi_classes].name,buffer_aux);

            buffer_aux = strtok(NULL,",");
            buffer_aux = strtok(NULL,",");
            fi_classes[pred_comp.fi_classes].accuracy = atoi(buffer_aux);

            buffer_aux = strtok(NULL,",");
            fi_classes[pred_comp.fi_classes].left_x = atoi(buffer_aux);

            buffer_aux = strtok(NULL,",");
            fi_classes[pred_comp.fi_classes].top_y = atoi(buffer_aux);

            buffer_aux = strtok(NULL,",");
            fi_classes[pred_comp.fi_classes].width = atoi(buffer_aux);

            buffer_aux = strtok(NULL,",");
            fi_classes[pred_comp.fi_classes].height = atoi(buffer_aux);

            fi_classes[pred_comp.fi_classes].compared = 0u;
            // printf("name: %s\t accuracy: %d\t left_x: %d\t top_y:%d\t width:%d\t height:%d\n",fi_classes[pred_comp.fi_classes].name,fi_classes[pred_comp.fi_classes].accuracy, fi_classes[pred_comp.fi_classes].left_x,fi_classes[pred_comp.fi_classes].top_y,fi_classes[pred_comp.fi_classes].width,fi_classes[pred_comp.fi_classes].height);
            pred_comp.fi_classes++;
        }

        while(!feof(p_file_golden))
        {
            pred_comp.golden_classes++;
            fscanf(p_file_golden,"%[^\n] ",buffer);

            buffer_aux = strtok(buffer,"_");
            strcpy(golden_classes.name,buffer_aux);

            buffer_aux = strtok(NULL,",");
            buffer_aux = strtok(NULL,",");
            golden_classes.accuracy = atoi(buffer_aux);

            buffer_aux = strtok(NULL,",");
            golden_classes.left_x = atoi(buffer_aux);

            buffer_aux = strtok(NULL,",");
            golden_classes.top_y = atoi(buffer_aux);

            buffer_aux = strtok(NULL,",");
            golden_classes.width = atoi(buffer_aux);

            buffer_aux = strtok(NULL,",");
            golden_classes.height = atoi(buffer_aux);

            // printf("name: %s\t accuracy: %d\t left_x: %d\t top_y:%d\t width:%d\t height:%d\n",golden_classes.name,golden_classes.accuracy, golden_classes.left_x,golden_classes.top_y,golden_classes.width,golden_classes.height);

            // Compare fi predictions with golden predictions
            for (size_t idx_clasess = 0; idx_clasess < pred_comp.fi_classes; idx_clasess++)
            {
                // Verify if it has already been compared
                if(fi_classes[idx_clasess].compared == 0u)
                {
                    // Verify if both have the same class (name)
                    if (strcmp(fi_classes[idx_clasess].name,golden_classes.name)==0)
                    {
                        // Verify the point center (x-axis)
                        if ((abs(fi_classes[idx_clasess].left_x) - abs(golden_classes.left_x)) < left_x_range)
                        {
                            // Verify the point center (y-axis)
                            if ((abs(fi_classes[idx_clasess].top_y) - abs(golden_classes.top_y)) < top_y_range)
                            {
                                // Check the size of the box (x-axis)
                                if ((abs(fi_classes[idx_clasess].width) - abs(golden_classes.width)) < width_range)
                                {
                                    // Check the size of the box (y-axis)
                                    if ((abs(fi_classes[idx_clasess].height) - abs(golden_classes.height)) < heigth_range)
                                    {
                                        // Check the accuracy
                                        if ((abs(fi_classes[idx_clasess].accuracy) - abs(golden_classes.accuracy)) < accuracy_range)
                                        {
                                            
                                            fi_classes[idx_clasess].compared = 1u;
                                            pred_comp.expected_detection++;
                                            break;
                                        }
                                        else if (fi_classes[idx_clasess].accuracy > golden_classes.accuracy)
                                        {
                                            fi_classes[idx_clasess].compared = 1u;
                                            pred_comp.expected_detection++;
                                            break;
                                        }
                                        else
                                        {
                                            fi_classes[idx_clasess].compared = 1u;
                                            pred_comp.misclassification_accuracy++;
                                            break;
                                        }
                                        
                                    }
                                    else
                                    {
                                        pred_comp.misclassification_height++;
                                        fi_classes[idx_clasess].compared = 1u;
                                        if (idx_clasess==(pred_comp.fi_classes-1))
                                        {
                                            pred_comp.undetected_classes++;
                                            break;
                                        }
                                    }
                                    
                                }
                                else
                                {
                                    // Nothing
                                    if (idx_clasess==(pred_comp.fi_classes-1))
                                    {
                                        pred_comp.undetected_classes++;
                                        break;
                                    }
                                }
                                
                            }
                            else
                            {
                                // Nothing
                                if (idx_clasess==(pred_comp.fi_classes-1))
                                {
                                    pred_comp.undetected_classes++;
                                    break;
                                }
                            }
                            
                        }
                        else
                        {
                            if (idx_clasess==(pred_comp.fi_classes-1))
                            {
                                pred_comp.undetected_classes++;
                                break;
                            }
                        }
                    }
                }
                else
                {
                    // Nothing
                }
            }
        }
        // Store the results in a .csv file
        fprintf(p_file_comparison,"%s,%d,%d,%d,%d,%d,%d,%d\n",str_file_name, pred_comp.golden_classes, pred_comp.fi_classes,
            pred_comp.expected_detection, pred_comp.misclassification_accuracy, pred_comp.misclassification_height,
            pred_comp.misclassification_widht, pred_comp.undetected_classes);   
        fclose(p_file_fi_results);
    }
    fclose(p_file_list);
    fclose(p_file_golden);
    fclose(p_file_comparison);
    
    return(0);
}
