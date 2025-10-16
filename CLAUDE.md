AITA llm automatic grading system

1. smart grouping all image to student, match name with a student roster
   1. use easy ocr
   2. crop the top n percent of the image for the name part
   3. fuzzy match with the provided name list, say elizabaath could match with elizabeth in the student name roster
   4. after extract the student name, because the test is standard, thus all student will have the same number of pages. say 5, you just need to sort the image by creation time, and then group every5 and then create a folder to orgnzie them, the folder name will be the name you extracted before.
2. question extraction, points and question body and question name per page recorded
   1. extract each question body and points that question have and the question header for example question 1a, 1b or 2c based on only one student's image
   2. use the google cloud storeage, it is a public bucket, send image url to llm api openrouter google/gemini-2.5-flash
   3. orgnized it well to reconstruct the exam
3. rubric and key generation for grading
   1. genaerte answer to each question you extracted in step2, one by one using llm. save it
   2. generate rubric for each question
   3. user can provide their own rubric for each question if no then llm genearte it based on instruction, if no instruciton then llm generate it
4. transcription per image for each student to text
   1. use the gcs as before, the same module, transcribe all image of all students. save it in a folder called transcription results
5. grading with llm
   1. grading by llm page by page, for each page, first grabe the corresponding rubric for all question in the page, and then use llm to grade it. record the score
   2. generate the grading reports.

save all the by product file here C:\Users\ellio\OneDrive - UW-Madison\AITA\intermediateproduct
save all the test file here C:\Users\ellio\OneDrive - UW-Madison\AITA\test
save all the doc file here C:\Users\ellio\OneDrive - UW-Madison\AITA\docs