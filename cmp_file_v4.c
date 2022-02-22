#include <stdio.h>
#include <string.h>
#include <stdlib.h> // atoi()
#include <math.h>   // abs()

#define SUMARIZE_EXP       0
#define UNDETECTED_EXP     1
#define MISCLASSIFIED_EXP  2
#define UNIFIED_EXP        3
#define CURRENT_EXP        SUMARIZE_EXP

struct Classes {
    char        name[100];
    __int32_t  accuracy;
    __int32_t  left_x;
    __int32_t  top_y;
    __int32_t  width;
    __int32_t  height;
    __int32_t  compared;
};

struct Pred_comp {
    __uint32_t golden_classes;
    __uint32_t expected_detection;
    __uint32_t fi_classes;
    __uint32_t misclassification_accuracy;
    __uint32_t misclassification_widht;
    __uint32_t misclassification_height;
    __int32_t undetected;
    __int32_t misclassified;
    __int32_t unified_comp; // this is a sum of undetected and misclassified
};

#define MAX_FILES_NUMBER 101u
#define MAX_STRING_SIZE 100u

struct Highest_misclassification {
    char file_names[MAX_STRING_SIZE];
    struct Pred_comp values;
};

int main () {
    // Definition of the files involved in the comparison
    FILE *p_file_list;
    FILE *p_file_golden;
    FILE *p_file_fi_results;
    FILE *p_file_comparison;
    FILE *p_file_undetect_comparison;
    FILE *p_file_misclass_comparison;
    FILE *p_file_unified_comparison;

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
    p_file_list = fopen("/home/javifdez/git_repositories/darknet/data/file_list_2.txt", "r");
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

    // Struct to store the number of classes detected:
    struct Pred_comp pred_comp;
    struct Classes golden_classes;
    struct Classes fi_classes[50u];

    struct Highest_misclassification high_undetec_comp[MAX_FILES_NUMBER];   // struct to store the files with the greatest undetected classes number sorted in descending order
    struct Highest_misclassification high_misclas_comp[MAX_FILES_NUMBER];   // struct to store the files with the greatest misclassification number sorted in descending order
    struct Highest_misclassification high_unified_comp[MAX_FILES_NUMBER];   // struct to store the files with the greatest unified number (undetected + misclassified) sorted in descending order

    // Auxiliar variables that define:
    // a) Number total of images (from 10000) that are not completly correct (or as expected) 
    __uint32_t ui32_qty_correct_images = 0u;
    __uint32_t ui32_qty_almost_correct_images = 0u;

    for (size_t idx32_init_high_comp = 0; idx32_init_high_comp < MAX_FILES_NUMBER; idx32_init_high_comp++)
    {
        high_undetec_comp[idx32_init_high_comp].values.undetected = 0;
        high_misclas_comp[idx32_init_high_comp].values.misclassified = 0;
        high_unified_comp[idx32_init_high_comp].values.unified_comp = 0;
    }
    
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
        snprintf(str_golden_file_name, 100,"/home/javifdez/git_repositories/darknet/results/golden_values/%s",str_file_name);

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
        // Store in struct "classes" the detected classes (from golden and from fi)
        // Move the pointer to the beggining of the file
        fseek(p_file_golden,0,SEEK_SET);
        fseek(p_file_fi_results,0,SEEK_SET);

        if (idx_loop==0)
        {
            fprintf(p_file_comparison,"image_name,golden_classes(exp),detected_classes,correct_detections,accuracy (mis),width (misc),height (miss),undetected,misclassifed,mis+und\n");
        }
        
        
        // Skip the first line
        fscanf(p_file_golden,"%[^\n] ",buffer);
        fscanf(p_file_fi_results,"%[^\n] ",buffer);

        // Initialize to 0 all values in pred_comp
        pred_comp.misclassification_widht    = 0u;      // Falla la anchura
        pred_comp.misclassification_height   = 0u;      // falla la altura
        pred_comp.misclassification_accuracy = 0u;      // falla la precisión
        pred_comp.golden_classes             = 0u;      // Número de detecciones en golden
        pred_comp.fi_classes                 = 0u;      // Número de detecciones en fi_result
        pred_comp.expected_detection         = 0u;      // Número de detecciones correctas
        pred_comp.undetected                 = 0;       // Clases de golden no detectadas en fi_results (golden_detection - expected_detection)
        pred_comp.misclassified              = 0;       // Classificación diferente a la golden  (fi_results - expected_detection)
        pred_comp.unified_comp               = 0;       // Suma de undetected and misclassified (undetected + misclassified = (fi_results - expected_detection) + (golden_detection - expected_detection) )

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
                                        // if (idx_clasess==(pred_comp.fi_classes-1))
                                        // {
                                        //     pred_comp.undetected++;
                                        //     break;
                                        // }
                                    }
                                    
                                }
                                else
                                {
                                    // Nothing
                                    // if (idx_clasess==(pred_comp.fi_classes-1))
                                    // {
                                    //     pred_comp.undetected++;
                                    //     break;
                                    // }
                                }
                                
                            }
                            else
                            {
                                // Nothing
                                // if (idx_clasess==(pred_comp.fi_classes-1))
                                // {
                                //     pred_comp.undetected++;
                                //     break;
                                // }
                            }
                            
                        }
                        else
                        {
                            // if (idx_clasess==(pred_comp.fi_classes-1))
                            // {
                            //     pred_comp.undetected++;
                            //     break;
                            // }
                        }
                    }
                }
                else
                {
                    // Nothing
                }
            }
        }
        pred_comp.undetected    = pred_comp.golden_classes - pred_comp.expected_detection;
        pred_comp.misclassified = pred_comp.fi_classes - pred_comp.expected_detection;
        pred_comp.unified_comp  = pred_comp.undetected + pred_comp.misclassified;


        // Store the results in a .csv file
        #if (SUMARIZE_EXP==CURRENT_EXP)
            fprintf(p_file_comparison,"%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",str_file_name, pred_comp.golden_classes, pred_comp.fi_classes,
                pred_comp.expected_detection, pred_comp.misclassification_accuracy, pred_comp.misclassification_height,
                pred_comp.misclassification_widht, pred_comp.undetected, pred_comp.misclassified, pred_comp.unified_comp);   
        #endif

        if (pred_comp.undetected==0u)
        {
            if(pred_comp.misclassified==0u)
            {
                ui32_qty_correct_images++;
            }
            else
            {
                ui32_qty_almost_correct_images++;
            }
        }

        if(idx_loop==(ui32_num_list_names-1))
        {
            printf("Correct predictions:%d in porcentage:%d",ui32_qty_correct_images,(ui32_qty_correct_images/ui32_num_list_names));
            printf("Correct predictions with new unexpected predictions:%d in porcentage:%d",ui32_qty_almost_correct_images,(ui32_qty_almost_correct_images/ui32_num_list_names));
        }
        

        // =============================================================================================================================
        //  Adequate the data to an easier interpretacion. The following code select the 100 most suitable images according to 3 criteria:
        //      1) The greatest number of undetected classes
        //      2) The greatest number of misclassification classes
        //      3) The greatest number of unified classes being unified the sum of undetected and misclassification classes
        // ============================================================================================================================= 
        // Create 3 auxiliar buffers to store the name of the image/file
        // Copy the name into the buffer
        // Create 3 auxiliar structs
        // Assign the current values of the struct to all the auxiliar values
        // #if (UNDETECTED_EXP==CURRENT_EXP)
        //     char aux_buffer_1[100u];
        //     strcpy(aux_buffer_1,str_file_name);
        //     struct Pred_comp pred_undet_comp;
        //     pred_undet_comp   = pred_comp;
        // #elif (MISCLASSIFIED_EXP==CURRENT_EXP)
        //     char aux_buffer_2[100u];
        //     strcpy(aux_buffer_2,str_file_name);
        //     struct Pred_comp pred_misclas_comp;
        //     pred_misclas_comp = pred_comp;
        // #else
        //     char aux_buffer_3[100u];
        //     strcpy(aux_buffer_3,str_file_name);
        //     struct Pred_comp pred_unified_comp;
        //     pred_unified_comp = pred_comp;
        // #endif
        
        // // Obtain a list in descendant order of the images with the highest number of undetected classe
        // if (idx_loop==0u)
        // {
        //     #if (UNDETECTED_EXP==CURRENT_EXP)
        //         strcpy(high_undetec_comp[0u].file_names,aux_buffer_1);
        //         high_undetec_comp[0u].values = pred_undet_comp;
        //     #elif (MISCLASSIFIED_EXP==CURRENT_EXP)
        //         strcpy(high_misclas_comp[0u].file_names,aux_buffer_2);
        //         high_misclas_comp[0u].values = pred_misclas_comp;
        //     #else
        //         strcpy(high_unified_comp[0u].file_names,aux_buffer_3);
        //         high_unified_comp[0u].values = pred_unified_comp;
        //     #endif
        // }
        // else 
        // {
        //     #if (UNDETECTED_EXP==CURRENT_EXP)
        //         // UNDETECTED CODE
        //         for (size_t idx32_var_1 = 0; idx32_var_1 < (MAX_FILES_NUMBER-1); idx32_var_1++)
        //         {
        //             // Verify if is greater than the upto now greatest value
        //             if(pred_undet_comp.undetected>high_undetec_comp[idx32_var_1].values.undetected)
        //             {
        //                 // Save the current Value in the last position of the array (It is an aux address)
        //                 high_undetec_comp[MAX_FILES_NUMBER]   = high_undetec_comp[idx32_var_1];

        //                 // Store in that position the new maximun value
        //                 high_undetec_comp[idx32_var_1].values = pred_undet_comp;
        //                 strcpy(high_undetec_comp[idx32_var_1].file_names,aux_buffer_1);

        //                 // To through the array and shift all the values and names 
        //                 size_t idx32_var_2 = 0u;
        //                 for (idx32_var_2 =(idx32_var_1+1u); idx32_var_2 < (MAX_FILES_NUMBER-2u); idx32_var_2++)
        //                 {
        //                     // We employ "pred_undet_comp" as auxiliar variable
        //                     pred_undet_comp = high_undetec_comp[MAX_FILES_NUMBER].values;
        //                     strcpy(aux_buffer_1, high_undetec_comp[MAX_FILES_NUMBER].file_names);
        //                     high_undetec_comp[MAX_FILES_NUMBER] = high_undetec_comp[idx32_var_2];
        //                     high_undetec_comp[idx32_var_2].values = pred_undet_comp;
        //                     strcpy(high_undetec_comp[idx32_var_2].file_names,aux_buffer_1);
        //                 }
        //                 break;
        //             } else
        //             {
        //                 // Nothing
        //             }
        //         }
        //     #elif (MISCLASSIFIED_EXP==CURRENT_EXP)
        //         // MISCLASSIFIED CODE
        //         __uint32_t idx32_var_3;
        //         for (idx32_var_3 = 0; idx32_var_3 < (MAX_FILES_NUMBER-1); idx32_var_3++)
        //         {
        //             // Verify if is greater than the upto now greatest value
        //             if(pred_misclas_comp.misclassified > high_misclas_comp[idx32_var_3].values.misclassified)
        //             {
        //                 // Save the current Value in the last position of the array (It is an aux address)
        //                 high_misclas_comp[MAX_FILES_NUMBER] = high_misclas_comp[idx32_var_3];

        //                 // Store in that position the new maximun value
        //                 high_misclas_comp[idx32_var_3].values = pred_misclas_comp;
        //                 strcpy(high_misclas_comp[idx32_var_3].file_names,aux_buffer_2);

        //                 // To through the array and shift all the values and names 
        //                 size_t idx32_var_4 = 0u;
        //                 for (idx32_var_4 =(idx32_var_3+1u); idx32_var_4 < (MAX_FILES_NUMBER-2u); idx32_var_4++)
        //                 {
        //                     // We employ "pred_misclas_comp" as auxiliar variable
        //                     pred_misclas_comp = high_misclas_comp[MAX_FILES_NUMBER].values;
        //                     strcpy(aux_buffer_2, high_misclas_comp[MAX_FILES_NUMBER].file_names);
        //                     high_misclas_comp[MAX_FILES_NUMBER] = high_misclas_comp[idx32_var_4];
        //                     high_misclas_comp[idx32_var_4].values = pred_misclas_comp;
        //                     strcpy(high_misclas_comp[idx32_var_4].file_names,aux_buffer_2);
        //                 }
        //                 idx32_var_3=MAX_FILES_NUMBER;
        //             }
        //         }
        //     #else
        //         // UNDETECTED CODE
        //         for (size_t idx32_var_1 = 0; idx32_var_1 < (MAX_FILES_NUMBER-1); idx32_var_1++)
        //         {
        //             // Verify if is greater than the upto now greatest value
        //             if(pred_unified_comp.unified_comp>high_unified_comp[idx32_var_1].values.unified_comp)
        //             {
        //                 // Save the current Value in the last position of the array (It is an aux address)
        //                 high_unified_comp[MAX_FILES_NUMBER]   = high_unified_comp[idx32_var_1];

        //                 // Store in that position the new maximun value
        //                 high_unified_comp[idx32_var_1].values = pred_unified_comp;
        //                 strcpy(high_unified_comp[idx32_var_1].file_names,aux_buffer_3);

        //                 // To through the array and shift all the values and names 
        //                 size_t idx32_var_2 = 0u;
        //                 for (idx32_var_2 =(idx32_var_1+1u); idx32_var_2 < (MAX_FILES_NUMBER-2u); idx32_var_2++)
        //                 {
        //                     // We employ "pred_unified_comp" as auxiliar variable
        //                     pred_unified_comp = high_unified_comp[MAX_FILES_NUMBER].values;
        //                     strcpy(aux_buffer_3, high_unified_comp[MAX_FILES_NUMBER].file_names);
        //                     high_unified_comp[MAX_FILES_NUMBER] = high_unified_comp[idx32_var_2];
        //                     high_unified_comp[idx32_var_2].values = pred_unified_comp;
        //                     strcpy(high_unified_comp[idx32_var_2].file_names,aux_buffer_3);
        //                 }
        //                 break;
        //             } else
        //             {
        //                 // Nothing
        //             }
        //         }
        //     #endif
        // }
 
        fclose(p_file_fi_results);
    }
    // =====================================================  p_file_misclass_comparison p_file_unified_comparison 
    //  Open the files and verify that can be opened
    // =====================================================
    // #if (UNDETECTED_EXP==CURRENT_EXP)
    //     p_file_undetect_comparison = fopen("max_undetected_cmp.csv", "w");
    //     if (p_file_undetect_comparison == NULL) 
    //     {
    //         printf("Could not open file max_undetected_cmp.csv\n");
    //         return (0u); 
    //     }

    // #elif (MISCLASSIFIED_EXP==CURRENT_EXP)
    //     p_file_misclass_comparison = fopen("max_misclass_cmp.csv", "w");
    //     if (p_file_misclass_comparison == NULL) 
    //     {
    //         printf("Could not open file max_undetected_comp.csv\n");
    //         return (0u); 
    //     }
    // #else
    //     p_file_unified_comparison = fopen("max_unified_cmp.csv", "w");
    //     if (p_file_unified_comparison == NULL) 
    //     {
    //         printf("Could not open file max_unified_comp.csv\n");
    //         return (0u); 
    //     }
    // #endif


    // #if (UNDETECTED_EXP==CURRENT_EXP)
    //     fprintf(p_file_undetect_comparison,"image_name,golden_classes(exp),detected_classes,correct_detections,accuracy (mis),width (misc),height (miss),undetected,misclassifed,mis+und\n");
    // #elif (MISCLASSIFIED_EXP==CURRENT_EXP)
    //     fprintf(p_file_misclass_comparison,"image_name,golden_classes(exp),detected_classes,correct_detections,accuracy (mis),width (misc),height (miss),undetected,misclassifed,mis+und\n");
    // #else    
    //     fprintf(p_file_unified_comparison,"image_name,golden_classes(exp),detected_classes,correct_detections,accuracy (mis),width (misc),height (miss),undetected,misclassifed,mis+und\n");
    // #endif

    // for (size_t idx32_aux_loop = 0; idx32_aux_loop < (MAX_FILES_NUMBER-1); idx32_aux_loop++)
    // {
    //     // Store the results in csv fileS
    //     #if (UNDETECTED_EXP==CURRENT_EXP)
    //         fprintf(p_file_undetect_comparison,"%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", high_undetec_comp[idx32_aux_loop].file_names, high_undetec_comp[idx32_aux_loop].values.golden_classes, high_undetec_comp[idx32_aux_loop].values.fi_classes,
    //             high_undetec_comp[idx32_aux_loop].values.expected_detection, high_undetec_comp[idx32_aux_loop].values.misclassification_accuracy, high_undetec_comp[idx32_aux_loop].values.misclassification_height,
    //             high_undetec_comp[idx32_aux_loop].values.misclassification_widht, high_undetec_comp[idx32_aux_loop].values.undetected, high_undetec_comp[idx32_aux_loop].values.misclassified, high_undetec_comp[idx32_aux_loop].values.unified_comp);   
    //     #elif (MISCLASSIFIED_EXP==CURRENT_EXP)
    //         fprintf(p_file_misclass_comparison,"%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", high_misclas_comp[idx32_aux_loop].file_names, high_misclas_comp[idx32_aux_loop].values.golden_classes, high_misclas_comp[idx32_aux_loop].values.fi_classes,
    //             high_misclas_comp[idx32_aux_loop].values.expected_detection, high_misclas_comp[idx32_aux_loop].values.misclassification_accuracy, high_misclas_comp[idx32_aux_loop].values.misclassification_height,
    //             high_misclas_comp[idx32_aux_loop].values.misclassification_widht, high_misclas_comp[idx32_aux_loop].values.undetected, high_misclas_comp[idx32_aux_loop].values.misclassified, high_misclas_comp[idx32_aux_loop].values.unified_comp);   
    //     #else
    //         fprintf(p_file_unified_comparison,"%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", high_unified_comp[idx32_aux_loop].file_names, high_unified_comp[idx32_aux_loop].values.golden_classes, high_unified_comp[idx32_aux_loop].values.fi_classes,
    //             high_unified_comp[idx32_aux_loop].values.expected_detection, high_unified_comp[idx32_aux_loop].values.misclassification_accuracy, high_unified_comp[idx32_aux_loop].values.misclassification_height,
    //             high_unified_comp[idx32_aux_loop].values.misclassification_widht, high_unified_comp[idx32_aux_loop].values.undetected, high_unified_comp[idx32_aux_loop].values.misclassified, high_unified_comp[idx32_aux_loop].values.unified_comp);   
    //     #endif
    // }
    
    // #if (UNDETECTED_EXP==CURRENT_EXP)
    //     fclose(p_file_undetect_comparison);
    // #elif (MISCLASSIFIED_EXP==CURRENT_EXP)
    //     fclose(p_file_misclass_comparison);
    // #else
    //     fclose(p_file_unified_comparison);
    // #endif

    fclose(p_file_list);
    fclose(p_file_golden);
    fclose(p_file_comparison);
    
    return(0);
}
